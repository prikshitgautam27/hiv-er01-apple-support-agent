"""
eval_harness.py
------------------
Scores the drafted replies in pipeline_output.csv on quality dimensions
using an LLM-as-judge, and provides the machinery to check how well
that judge agrees with a human -- required by the assignment.

Two-part workflow:
  1. Run this script: it judges every reply on 3 dimensions (1-5 scale):
     relevance, groundedness, tone. Produces judge_scores.csv.
  2. YOU manually score a random subset (default 25) of the SAME rows
     yourself (a simple prompt walks you through it), producing
     human_scores.csv.
  3. Run this script again with --correlate to compute Spearman
     correlation between your scores and the judge's -- this is your
     "evidence of how well your judge agrees with a human" deliverable.

Run: python eval_harness.py              (judges all replies)
     python eval_harness.py --human-sample 25   (interactive human scoring)
     python eval_harness.py --correlate         (compute agreement)
"""

import os
import sys
import time
import json
import polars as pl
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field
from scipy.stats import spearmanr

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise SystemExit("GROQ_API_KEY not found in .env")

PIPELINE_OUTPUT_FILE = "pipeline_output.csv"
JUDGE_SCORES_FILE = "judge_scores.csv"
HUMAN_SCORES_FILE = "human_scores.csv"
MODEL = "openai/gpt-oss-20b"  # lighter model for judging -- separate quota, plenty for a rubric task

JUDGE_SYSTEM_PROMPT = """You are grading a customer support reply for quality.
You will see: the customer's message, the historical precedent the reply was grounded in, and the drafted reply.

Score the reply on three dimensions, each 1-5 (5 = best):
- relevance: does the reply actually address what the customer asked?
- groundedness: is the reply consistent with the historical precedent shown, not making things up?
- tone: is the tone appropriate, professional, and empathetic?

Respond with ONLY this JSON shape, no other text:
{"relevance": <1-5>, "groundedness": <1-5>, "tone": <1-5>}
"""


class JudgeScore(BaseModel):
    relevance: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    tone: int = Field(ge=1, le=5)


from api_cache import cached_call


def judge_reply(customer_text: str, precedent: str, reply: str, max_retries: int = 4):
    user_msg = f"Customer message: {customer_text}\n\nHistorical precedent: {precedent}\n\nDrafted reply: {reply}"

    cache_key = f"{MODEL}|judge|{user_msg}"
    cached = cached_call(cache_key)
    if cached is not None:
        try:
            return JudgeScore(**json.loads(cached))
        except (json.JSONDecodeError, ValidationError):
            pass  # fall through and re-request if cached value was somehow bad

    for attempt in range(max_retries):
        start_time = time.time()
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": 300,  # was 100 -- too tight, caused truncated/malformed JSON
                    "reasoning_effort": "low",
                    "temperature": 0,
                },
                timeout=30,
            )
            if response.status_code == 429:
                wait = float(response.headers.get("retry-after", 5 * (2 ** attempt)))
                print(f"   [rate limit] waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"].strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            score = JudgeScore(**json.loads(raw))
            cached_call(cache_key, save=json.dumps(score.model_dump()))  # cache success only

            # Adaptive pacing: sleep only the leftover of the target window,
            # not a fixed delay -- keeps runtime close to the rate-limit
            # boundary (~7 min for 200 rows) instead of over-sleeping.
            elapsed = time.time() - start_time
            time.sleep(max(0.1, 2.1 - elapsed))
            return score
        except (requests.RequestException, json.JSONDecodeError, ValidationError, KeyError) as e:
            print(f"   [warning] judge call failed (attempt {attempt + 1}): {e}")
            time.sleep(2)
    return None


def run_judge(sample_size: int = 0):
    print(f"Loading {PIPELINE_OUTPUT_FILE}...")
    df = pl.read_csv(PIPELINE_OUTPUT_FILE)
    if sample_size and sample_size < df.height:
        print(f"Using a {sample_size}-row sample for a quick reproduction run.")
        df = df.sample(n=sample_size, seed=42)

    relevance, groundedness, tone = [], [], []
    for i, row in enumerate(df.iter_rows(named=True)):
        print(f"  Judging {i + 1}/{df.height}...", end="\r")
        score = judge_reply(row["customer_text"], row.get("retrieved_precedent", ""), row["drafted_reply"])
        if score:
            relevance.append(score.relevance)
            groundedness.append(score.groundedness)
            tone.append(score.tone)
        else:
            relevance.append(None)
            groundedness.append(None)
            tone.append(None)
        time.sleep(2.2)

    print(f"\n\nDone. Average scores: relevance={sum(r for r in relevance if r)/len(relevance):.2f}, "
          f"groundedness={sum(g for g in groundedness if g)/len(groundedness):.2f}, "
          f"tone={sum(t for t in tone if t)/len(tone):.2f}")

    result = df.with_columns(
        [
            pl.Series("judge_relevance", relevance),
            pl.Series("judge_groundedness", groundedness),
            pl.Series("judge_tone", tone),
        ]
    )
    result.write_csv(JUDGE_SCORES_FILE)
    print(f"Wrote {JUDGE_SCORES_FILE}")
    print(f"\nNext: run 'python eval_harness.py --human-sample 25' to score a subset yourself,")
    print("then 'python eval_harness.py --correlate' to check judge-human agreement.")


def run_human_sample(n: int):
    df = pl.read_csv(JUDGE_SCORES_FILE)

    # Stratified sampling: pure random sampling mostly picks already-good
    # replies (since most outputs score 4-5), which produces near-zero
    # score variance and makes correlation mathematically undetectable
    # even if judge and human broadly agree. Deliberately include some
    # of the judge's LOWEST-scored replies so there's real spread to
    # correlate against.
    df = df.with_columns(
        ((pl.col("judge_relevance") + pl.col("judge_groundedness") + pl.col("judge_tone")) / 3).alias("_avg_judge_score")
    )
    df_sorted = df.sort("_avg_judge_score")

    n_low = n // 3
    n_high = n // 3
    n_random = n - n_low - n_high

    low_scored = df_sorted.head(max(n_low, 1))
    high_scored = df_sorted.tail(max(n_high, 1))
    remaining = df_sorted.filter(~pl.col("customer_tweet_id").is_in(
        pl.concat([low_scored["customer_tweet_id"], high_scored["customer_tweet_id"]])
    ))
    random_middle = remaining.sample(n=min(n_random, remaining.height), seed=42)

    sample = pl.concat([low_scored, high_scored, random_middle]).unique(subset=["customer_tweet_id"]).sample(fraction=1.0, shuffle=True, seed=42)

    print(f"You'll score {sample.height} replies on the same 3 dimensions (1-5) the judge used.")
    print("Sample is stratified (some low-, some high-scored by the judge) to ensure enough")
    print("score variance for a meaningful correlation -- pure random sampling tends to only")
    print("pick already-good replies, making agreement mathematically undetectable.")
    print("Press Enter to accept a default of 3 for any dimension.\n")

    human_relevance, human_groundedness, human_tone, tweet_ids = [], [], [], []
    for row in sample.iter_rows(named=True):
        print(f"\n{'=' * 70}")
        print(f"Customer: {row['customer_text']}")
        print(f"Drafted reply: {row['drafted_reply']}")
        print(f"(Judge scored this: relevance={row['judge_relevance']}, groundedness={row['judge_groundedness']}, tone={row['judge_tone']})")

        def ask(dim):
            while True:
                raw = input(f"Your {dim} score (1-5) > ").strip()
                if raw == "":
                    return 3
                if raw.isdigit() and 1 <= int(raw) <= 5:
                    return int(raw)
                print("  Enter a number 1-5.")

        human_relevance.append(ask("relevance"))
        human_groundedness.append(ask("groundedness"))
        human_tone.append(ask("tone"))
        tweet_ids.append(row["customer_tweet_id"])

    out = pl.DataFrame({
        "customer_tweet_id": tweet_ids,
        "human_relevance": human_relevance,
        "human_groundedness": human_groundedness,
        "human_tone": human_tone,
    })
    out.write_csv(HUMAN_SCORES_FILE)
    print(f"\nWrote {HUMAN_SCORES_FILE}. Run 'python eval_harness.py --correlate' next.")


def run_correlation():
    judge_df = pl.read_csv(JUDGE_SCORES_FILE)
    human_df = pl.read_csv(HUMAN_SCORES_FILE)
    merged = human_df.join(judge_df, on="customer_tweet_id", how="inner")

    print(f"Comparing {merged.height} human-scored rows against judge scores...\n")

    for dim in ["relevance", "groundedness", "tone"]:
        human_vals = merged[f"human_{dim}"].to_list()
        judge_vals = merged[f"judge_{dim}"].to_list()

        corr, p_value = spearmanr(human_vals, judge_vals)
        exact_match = sum(1 for h, j in zip(human_vals, judge_vals) if h == j) / len(human_vals)
        within_one = sum(1 for h, j in zip(human_vals, judge_vals) if abs(h - j) <= 1) / len(human_vals)

        print(f"{dim}:")
        print(f"   Spearman correlation = {corr:.3f} (p={p_value:.3f})")
        print(f"   Exact match: {exact_match:.1%}  |  Within 1 point: {within_one:.1%}")
        print(f"   Human score spread: min={min(human_vals)}, max={max(human_vals)}, distinct values={len(set(human_vals))}")
        print(f"   Judge score spread: min={min(judge_vals)}, max={max(judge_vals)}, distinct values={len(set(judge_vals))}")
        print()

    print("NOTE: if score spread above is narrow (e.g. only values 4-5 appear), Spearman")
    print("correlation is mathematically unreliable regardless of true agreement -- this is")
    print("a ceiling effect, not necessarily judge unreliability. Report 'within 1 point' and")
    print("exact match alongside Spearman for a fuller, more honest picture in your writeup.")


if __name__ == "__main__":
    if "--human-sample" in sys.argv:
        idx = sys.argv.index("--human-sample")
        n = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 25
        run_human_sample(n)
    elif "--correlate" in sys.argv:
        run_correlation()
    elif "--sample" in sys.argv:
        idx = sys.argv.index("--sample")
        n = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 20
        run_judge(sample_size=n)
    else:
        run_judge()
