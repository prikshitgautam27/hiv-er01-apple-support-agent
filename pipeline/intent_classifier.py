"""
intent_classifier.py
----------------------
Classifies customer tweets into one of 6 intents using Groq, BATCHED
(multiple tweets per API call) to stay within rate limits and the
assignment's 15-minute reproduction budget.

Why batching matters: at Groq's free-tier 30 requests/minute limit,
classifying 200 tweets one-at-a-time takes ~7-8 minutes just for this
one step. Batching 10 tweets per call cuts that to under a minute.

Run: python intent_classifier.py
Requires: pip install polars pydantic requests python-dotenv scikit-learn
Output: classifier_predictions.csv + a printed classification report
"""

import os
import time
import json
import polars as pl
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing import Literal, List
from sklearn.metrics import classification_report, confusion_matrix

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise SystemExit("GROQ_API_KEY not found in .env")

GOLDEN_SET_FILE = "golden_set_final.csv"
OUTPUT_FILE = "classifier_predictions.csv"
MODEL = "openai/gpt-oss-120b"
BATCH_SIZE = 10  # tweets per API call -- tune down if you hit token limits

INTENTS = [
    "battery_power_drain",
    "ios_update_issues",
    "connectivity_hardware",
    "apple_id_account",
    "app_store_billing",
    "out_of_scope_or_other",
]

SYSTEM_PROMPT = f"""You are an intent classifier for AppleSupport customer tweets.

Classify EACH tweet into exactly one of these intents:
{', '.join(INTENTS)}

Definitions:
- battery_power_drain: battery life, charging, overheating, dies fast
- ios_update_issues: an update/upgrade caused a problem, or asks about update availability
- connectivity_hardware: WiFi/Bluetooth/signal issues, or physical hardware problems (screen, buttons, GPS)
- apple_id_account: login, password, locked account, Apple ID issues
- app_store_billing: wrong charge, refund request, subscription/payment issue
- out_of_scope_or_other: doesn't clearly fit above, mixes multiple issues, or is praise/spam

You will be given a numbered list of tweets. Respond with ONLY a JSON array,
one object per tweet, in the SAME ORDER, no other text:
[{{"id": 1, "intent": "..."}}, {{"id": 2, "intent": "..."}}, ...]
"""


class ClassificationItem(BaseModel):
    id: int
    intent: Literal[
        "battery_power_drain",
        "ios_update_issues",
        "connectivity_hardware",
        "apple_id_account",
        "app_store_billing",
        "out_of_scope_or_other",
    ]


def classify_batch(tweets: List[str], max_retries: int = 5) -> List[str]:
    """Classify a batch of tweets in a single API call. Returns a list of
    intents in the same order as the input, or 'FAILED' for any that
    couldn't be parsed/validated."""
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(tweets))

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": numbered},
                    ],
                    "max_tokens": 150 * len(tweets),  # scale budget with batch size
                    "reasoning_effort": "low",
                    "temperature": 0,
                },
                timeout=45,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("retry-after")
                wait = float(retry_after) if retry_after else (5 * (2 ** attempt))
                print(f"   [rate limit] waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"].strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)

            results = ["FAILED"] * len(tweets)
            for item in parsed:
                validated = ClassificationItem(**item)
                idx = validated.id - 1
                if 0 <= idx < len(tweets):
                    results[idx] = validated.intent
            return results

        except (requests.RequestException, json.JSONDecodeError, ValidationError, KeyError, TypeError) as e:
            print(f"   [warning] batch failed (attempt {attempt + 1}): {e}")
            time.sleep(3)

    return ["FAILED"] * len(tweets)


def main():
    print(f"Loading {GOLDEN_SET_FILE}...")
    df = pl.read_csv(GOLDEN_SET_FILE)
    texts = df["customer_text"].to_list()
    true_intents = df["true_intent"].to_list()

    predictions = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Classifying {len(texts)} tweets in {total_batches} batches of {BATCH_SIZE}...")
    for i in range(0, len(texts), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches}...", end="\r")
        batch = texts[i : i + BATCH_SIZE]
        results = classify_batch(batch)
        predictions.extend(results)
        time.sleep(2.2)  # stay under Groq's 30 RPM free-tier limit

    print(f"\n\nDone. {predictions.count('FAILED')} batches/items failed after retries.")

    # Save predictions alongside ground truth
    result_df = df.with_columns(pl.Series("predicted_intent", predictions))
    result_df.write_csv(OUTPUT_FILE)
    print(f"Wrote {OUTPUT_FILE}")

    # Evaluate against golden set (excluding any FAILED rows from the report)
    valid_mask = [p != "FAILED" for p in predictions]
    y_true = [t for t, v in zip(true_intents, valid_mask) if v]
    y_pred = [p for p, v in zip(predictions, valid_mask) if v]

    print(f"\n{'=' * 60}")
    print(f"CLASSIFICATION REPORT ({len(y_true)}/{len(texts)} rows scored)")
    print(f"{'=' * 60}")
    print(classification_report(y_true, y_pred, labels=INTENTS, zero_division=0))

    print("Confusion matrix (rows=true, columns=predicted):")
    print(f"Labels order: {INTENTS}")
    print(confusion_matrix(y_true, y_pred, labels=INTENTS))


if __name__ == "__main__":
    main()
