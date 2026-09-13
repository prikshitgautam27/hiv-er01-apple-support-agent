"""
full_pipeline.py
------------------
The complete agent: for each customer tweet, produces
  1. predicted_intent      (LLM classifier, batched)
  2. drafted_reply         (LLM, grounded in retrieved historical precedent)
  3. routing_decision      (auto_handle / escalate -- RULE-BASED, not LLM)
  4. escalation_reason     (only if escalate)

DESIGN DECISION (put this in your decision log): routing is decided by
deterministic rules, not the LLM. Rules are auditable, consistent, and
don't hallucinate a justification -- appropriate for the highest-stakes
decision in the pipeline. The LLM's job is classification and drafting,
where creative/pattern-matching ability actually helps.

Run: python full_pipeline.py [--sample N]
Requires: pip install polars pydantic requests python-dotenv chromadb sentence-transformers
Output: pipeline_output.csv
"""

import os
import re
import sys
import time
import json
import polars as pl
import requests
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing import Literal, List, Optional

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise SystemExit("GROQ_API_KEY not found in .env")

INPUT_FILE = "golden_set_final.csv"
OUTPUT_FILE = "pipeline_output.csv"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "apple_support_history"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MODEL = "openai/gpt-oss-20b"  # switched from 120b temporarily -- separate quota bucket, unblocks iteration
                                # while 120b's daily quota is exhausted. Use 120b again for your final,
                                # single, reportable full-dataset run once quota resets.
CLASSIFY_BATCH_SIZE = 10
TOP_K_RETRIEVAL = 3

# Default sample size for a fast, reproducible demo run (see README 15-min budget).
# Pass --sample 0 (or a larger number) to run the full file.
DEFAULT_SAMPLE_SIZE = 30

INTENTS = [
    "battery_power_drain",
    "ios_update_issues",
    "connectivity_hardware",
    "apple_id_account",
    "app_store_billing",
    "out_of_scope_or_other",
]

# ---------------------------------------------------------------------------
# STEP 0: Build few-shot examples programmatically from NON-golden data
# (avoids evaluation leakage -- these are never from golden_set_final.csv)
# ---------------------------------------------------------------------------

FEWSHOT_KEYWORDS = {
    "battery_power_drain": [r"\bbattery\b.{0,30}\b(drain|dies|dead|life|percent)\b"],
    "ios_update_issues": [r"\b(after|since)\b.{0,15}\b(update|upgrad)", r"\bupdate\b.{0,20}\b(broke|broken|issue|problem|bug|stuck|fail)"],
    "connectivity_hardware": [r"\b(wifi|bluetooth)\b", r"\bscreen\b.{0,15}\b(won't|doesn't|black|crack|broke)", r"\bbutton\b.{0,15}\b(won't|stuck|broke)"],
    "apple_id_account": [r"\b(apple id|password)\b.{0,20}\b(reset|forgot|locked|wrong|can't)"],
    "app_store_billing": [r"\b(charged|refund|billing|subscription)\b"],
    "out_of_scope_or_other": [r"\b(thanks|thank you|love (my|this|it))\b", r"^\s*@\w+\s+https?://"],
}


def build_fewshot_examples(pairs_file="apple_support_pairs.csv", excluded_ids_file="golden_excluded_ids.csv", n_per_intent=2):
    """Pulls a small number of high-confidence, unambiguous examples per intent
    from the historical pool, explicitly excluding golden-set tweet_ids so
    there is zero leakage between few-shot examples and evaluation data."""
    try:
        pairs = pl.read_csv(pairs_file)
        excluded = pl.read_csv(excluded_ids_file)
        pairs = pairs.join(excluded, on="customer_tweet_id", how="anti")
    except FileNotFoundError:
        return ""  # gracefully degrade to zero-shot if files aren't present

    examples = []
    for intent, patterns in FEWSHOT_KEYWORDS.items():
        combined_pattern = "|".join(patterns)
        matches = pairs.filter(pl.col("customer_text").str.contains(combined_pattern))
        picked = matches.head(n_per_intent)
        print(f"   [few-shot] {intent}: found {picked.height} matching example(s)")
        for row in picked.to_dicts():
            examples.append(f'Tweet: "{row["customer_text"][:150]}"\n-> intent: {intent}')

    return "\n\n".join(examples)


FEWSHOT_EXAMPLES = build_fewshot_examples()
if not FEWSHOT_EXAMPLES.strip():
    print("   [warning] No few-shot examples were found -- classifier is running effectively zero-shot!")

# ---------------------------------------------------------------------------
# STEP 1: Intent classification (batched LLM calls, few-shot + disambiguation)
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM_PROMPT = f"""You are an intent classifier for AppleSupport customer tweets.
Classify EACH tweet into exactly one of: {', '.join(INTENTS)}
Respond with ONLY a JSON array, one object per tweet, same order:
[{{"id": 1, "intent": "..."}}, ...]
"""
# NOTE: an earlier version of this prompt added few-shot examples and an
# explicit "classify by symptom not update mention" disambiguation rule.
# Measured on the full 200-row golden set, that version scored WORSE
# (76% vs 78% accuracy; ios_update_issues recall dropped 0.61->0.45).
# Reverted to this simpler prompt based on that evidence -- see report's
# failure analysis / decision log for the full comparison.


class ClassificationItem(BaseModel):
    id: int
    intent: Literal[
        "battery_power_drain", "ios_update_issues", "connectivity_hardware",
        "apple_id_account", "app_store_billing", "out_of_scope_or_other",
    ]


from api_cache import cached_call

REPLY_PACING_SECONDS = 2.1  # target window between live API calls (rate-limit safety margin)


def is_reply_malformed(text: str) -> bool:
    """Deterministic validation tier: catches truncated/malformed generations
    (e.g. cut off mid-sentence due to reasoning-token starvation) before they
    ever reach the output file, instead of silently shipping broken text."""
    if not text or len(text.strip()) < 15:
        return True
    stripped = text.strip()
    ends_clean = bool(re.search(r'[.!?"\u2019]$', stripped)) or bool(re.search(r'https?://\S+$', stripped))
    return not ends_clean


def call_groq(system_prompt: str, user_msg: str, max_tokens: int, max_retries: int = 5,
              reasoning_effort: str = "low", validator=None) -> Optional[str]:
    cache_key = f"{MODEL}|{reasoning_effort}|{system_prompt}|{user_msg}"
    cached = cached_call(cache_key)
    if cached is not None and (validator is None or validator(cached)):
        return cached  # instant, zero API calls -- only trust cache if it passes validation too

    for attempt in range(max_retries):
        start_time = time.time()
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": max_tokens,
                    "reasoning_effort": reasoning_effort,
                    "temperature": 0.3,
                },
                timeout=45,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("retry-after")
                wait = float(retry_after) if retry_after else (5 * (2 ** attempt))
                print(f"      [rate limit] waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"].strip()

            # Deterministic Output Validation Tier: reject and retry malformed
            # output instead of caching/returning something broken.
            if validator is not None and not validator(result):
                print(f"      [validation] output failed validation (attempt {attempt + 1}), retrying...")
                elapsed = time.time() - start_time
                time.sleep(max(0.1, REPLY_PACING_SECONDS - elapsed))
                continue

            cached_call(cache_key, save=result)  # only cache validated results

            # Adaptive pacing: sleep only the remainder of the target window,
            # not a fixed delay -- keeps total runtime close to the rate-limit
            # boundary instead of over-sleeping on fast responses.
            elapsed = time.time() - start_time
            time.sleep(max(0.1, REPLY_PACING_SECONDS - elapsed))
            return result
        except requests.RequestException as e:
            print(f"      [warning] call failed (attempt {attempt + 1}): {e}")
            time.sleep(3)
    return None


def classify_batch(tweets: List[str]) -> List[str]:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(tweets))
    raw = call_groq(CLASSIFY_SYSTEM_PROMPT, numbered, max_tokens=200 * len(tweets), reasoning_effort="low")
    results = ["out_of_scope_or_other"] * len(tweets)  # safe fallback, never crash the pipeline
    if raw is None:
        return results
    try:
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        for item in parsed:
            validated = ClassificationItem(**item)
            idx = validated.id - 1
            if 0 <= idx < len(tweets):
                results[idx] = validated.intent
    except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as e:
        print(f"      [warning] classification parse failed: {e}")
    return results


# ---------------------------------------------------------------------------
# STEP 2: Retrieval (local ChromaDB, no API cost)
# ---------------------------------------------------------------------------

def load_retriever():
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def retrieve_precedent(collection, tweet_text: str, k: int = TOP_K_RETRIEVAL):
    results = collection.query(query_texts=[tweet_text], n_results=k)
    precedents = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        precedents.append({"customer_text": doc, "apple_reply_text": meta["apple_reply_text"], "is_dm_redirect": meta["is_dm_redirect"]})
    return precedents


# ---------------------------------------------------------------------------
# STEP 3: Escalation guardrail -- RULE-BASED, deterministic, auditable
# ---------------------------------------------------------------------------

SAFETY_PATTERN = re.compile(r"\b(fire|smoke|explod|burn(ing)?|spark)\b", re.IGNORECASE)
SECURITY_PATTERN = re.compile(r"\b(hacked|locked out|compromis|stolen|scam|fraud|unauthorized)\b", re.IGNORECASE)
BILLING_PATTERN = re.compile(r"\b(refund|charged twice|overcharg|billing dispute|unauthorized charge)\b", re.IGNORECASE)
ANGRY_PATTERN = re.compile(r"\b(pissed|furious|terrible|worst|ridiculous|sue|lawyer|disgusted)\b", re.IGNORECASE)


NO_DM_PATTERN = re.compile(r"\bno need to dm\b|\bdon'?t (want|need) to dm\b|\bwithout dm\b", re.IGNORECASE)


def decide_routing(customer_text: str, predicted_intent: str, precedents: list) -> tuple[str, str]:
    """Returns (routing, reason). Deterministic rules, checked in priority order."""
    if SAFETY_PATTERN.search(customer_text):
        return "escalate", "safety_hazard"
    if SECURITY_PATTERN.search(customer_text):
        return "escalate", "account_security_risk"
    if BILLING_PATTERN.search(customer_text):
        return "escalate", "financial_refund_request"
    if ANGRY_PATTERN.search(customer_text):
        return "escalate", "angry_customer_needs_human_tone"

    # Respect explicit customer preference -- never escalate to DM if they said not to
    if NO_DM_PATTERN.search(customer_text):
        return "auto_handle", ""

    # Historical precedent signal: did Apple typically handle this via DM?
    # Raised threshold to 3/3 (was 2/3) -- at a 33% base rate of DM redirects
    # across the whole dataset, a 2/3 majority triggered too easily on benign
    # queries that only coincidentally retrieved DM-handled neighbors.
    dm_votes = sum(1 for p in precedents if p["is_dm_redirect"])
    if dm_votes == len(precedents) and len(precedents) > 0:
        return "escalate", "requires_dm_troubleshooting"

    return "auto_handle", ""


# ---------------------------------------------------------------------------
# STEP 4: Reply drafting -- grounded in retrieved precedent
# ---------------------------------------------------------------------------

def draft_reply(customer_text: str, predicted_intent: str, routing: str, precedents: list) -> str:
    examples = "\n\n".join(
        f"Past customer message: {p['customer_text']}\nApple's real reply: {p['apple_reply_text']}"
        for p in precedents
    )
    no_dm_note = ""
    if NO_DM_PATTERN.search(customer_text):
        no_dm_note = "IMPORTANT: the customer explicitly said they do not want to be asked to DM. Do NOT ask them to DM you. Address their issue directly in this reply instead.\n"

    if routing == "escalate":
        instruction = (
            "This message needs a human agent. Draft a SHORT, empathetic acknowledgment "
            "that confirms you understand the issue and are connecting them with the right team. "
            "Do not attempt to solve the issue yourself."
        )
    else:
        instruction = (
            "Draft a helpful, concise reply that follows the STYLE and troubleshooting pattern "
            "shown in the examples below, adapted to this specific customer's message. "
            "This has been decided as an AUTO-HANDLED case: you MUST resolve or give concrete "
            "next steps directly in this reply. Do NOT ask the customer to DM you, even if the "
            "examples below do so -- give the actual answer or troubleshooting step instead."
        )

    system_prompt = f"""You are drafting a reply as AppleSupport on Twitter.
The customer's message has intent: {predicted_intent}
{no_dm_note}
Historical examples of how Apple has responded to similar messages:
{examples}

{instruction}
Keep the reply under 280 characters, in Apple's typical tone (helpful, brief, professional).
Do NOT start the reply with an @mention or a tweet ID number -- start directly with the message.
Respond with ONLY the reply text, no other commentary.
"""
    raw = call_groq(system_prompt, customer_text, max_tokens=150, reasoning_effort="low", validator=lambda t: not is_reply_malformed(t))
    if not raw:
        return "[reply generation failed -- flagged for manual drafting]"
    # Strip zero-width/invisible unicode characters that occasionally leak into generation
    cleaned = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", raw)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    sample_size = DEFAULT_SAMPLE_SIZE
    if "--sample" in sys.argv:
        idx = sys.argv.index("--sample")
        sample_size = int(sys.argv[idx + 1])

    print(f"Loading {INPUT_FILE}...")
    df = pl.read_csv(INPUT_FILE)
    if sample_size and sample_size < df.height:
        print(f"Using a {sample_size}-row sample for this run (pass --sample 0 to run the full file).")
        df = df.sample(n=sample_size, seed=42)

    texts = df["customer_text"].to_list()
    tweet_ids = df["customer_tweet_id"].to_list()

    print("\nStep 1: Classifying intents (batched)...")
    predicted_intents = []
    for i in range(0, len(texts), CLASSIFY_BATCH_SIZE):
        batch = texts[i : i + CLASSIFY_BATCH_SIZE]
        print(f"  Batch {i // CLASSIFY_BATCH_SIZE + 1}/{(len(texts) + CLASSIFY_BATCH_SIZE - 1) // CLASSIFY_BATCH_SIZE}...", end="\r")
        predicted_intents.extend(classify_batch(batch))
    print()

    print("\nStep 2: Loading local RAG retriever...")
    collection = load_retriever()

    print("\nStep 3 & 4: Retrieving precedent, deciding routing, drafting replies...")
    routings, reasons, replies, precedent_summaries = [], [], [], []

    for i, (tweet_id, text, intent) in enumerate(zip(tweet_ids, texts, predicted_intents)):
        print(f"  Tweet {i + 1}/{len(texts)} (id={tweet_id})...", end="\r")
        precedents = retrieve_precedent(collection, text)
        routing, reason = decide_routing(text, intent, precedents)
        reply = draft_reply(text, intent, routing, precedents)

        routings.append(routing)
        reasons.append(reason)
        replies.append(reply)
        precedent_summaries.append("; ".join(p["customer_text"][:60] for p in precedents))

    print("\n\nDone. Writing results...")
    result_df = df.with_columns(
        [
            pl.Series("predicted_intent", predicted_intents),
            pl.Series("routing_decision", routings),
            pl.Series("escalation_reason_predicted", reasons),
            pl.Series("drafted_reply", replies),
            pl.Series("retrieved_precedent", precedent_summaries),
        ]
    )
    result_df.write_csv(OUTPUT_FILE)
    print(f"Wrote {OUTPUT_FILE} with {result_df.height} fully-processed tweets.")
    print("\nEach row now has: predicted_intent, routing_decision, escalation_reason_predicted, drafted_reply.")
    print("This is your complete agent output -- ready for the eval harness.")


if __name__ == "__main__":
    main()
