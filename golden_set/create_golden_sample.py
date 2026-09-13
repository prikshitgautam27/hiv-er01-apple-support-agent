"""
create_golden_sample.py
------------------------
Builds a STRATIFIED golden evaluation sample instead of pure random
sampling. Pure random sampling over-represents whatever is most common
in the data (dm_redirect / generic troubleshooting) and under-represents
rare-but-important intents (billing, account lockout, etc.), which makes
your eval set a weak test of the agent.

This script:
  1. Applies a cheap keyword heuristic to bucket every row into a rough
     intent (good enough for STRATIFICATION only -- you will still
     hand-label the true_intent column yourself).
  2. Samples an even-ish number of rows per bucket for the golden set.
  3. Writes the excluded golden_tweet_ids to a separate file so you can
     filter them out of your ChromaDB RAG index later (prevents
     train/eval leakage -- the agent should never retrieve the answer
     to a question it's being tested on).

Run: python create_golden_sample.py
Outputs:
  golden_set_template.csv   (150-250 rows, ready for hand-labeling)
  golden_excluded_ids.csv   (tweet_ids to exclude from the RAG index)
"""

import re
import polars as pl

INPUT_FILE = "apple_support_pairs.csv"
GOLDEN_OUTPUT = "golden_set_template.csv"
EXCLUDED_IDS_OUTPUT = "golden_excluded_ids.csv"

TARGET_GOLDEN_SIZE = 200  # pick anything in the 150-250 range
SEED = 42

# Cheap, transparent heuristic buckets -- ONLY used to stratify sampling,
# not as ground truth. You still hand-label true_intent yourself.
INTENT_KEYWORDS = {
    "battery_power_drain": r"\bbattery|drain|charg(e|ing)|power\b",
    "ios_update_issues": r"\bupdate|ios\s?\d|upgrad(e|ing)|install\b",
    "connectivity_hardware": r"\bwifi|bluetooth|connect|signal|screen|broken|crack|button\b",
    "apple_id_account": r"\bapple id|account|locked|password|login|sign in\b",
    "app_store_billing": r"\bcharged|refund|payment|subscription|billing|purchase|app store\b",
}


def bucket_intent(text: str) -> str:
    text = (text or "").lower()
    for intent, pattern in INTENT_KEYWORDS.items():
        if re.search(pattern, text):
            return intent
    return "out_of_scope_or_other"


print("Step 1: Loading prepared pairs...")
df = pl.read_csv(INPUT_FILE)

print("Step 2: Applying heuristic intent buckets (for stratification only)...")
df = df.with_columns(
    pl.col("customer_text").map_elements(bucket_intent, return_dtype=pl.Utf8).alias("_stratify_bucket")
)

buckets = df["_stratify_bucket"].unique().to_list()
per_bucket = max(1, TARGET_GOLDEN_SIZE // len(buckets))
print(f"Step 3: Sampling ~{per_bucket} rows per bucket across {len(buckets)} buckets: {buckets}")

sampled_parts = []
for bucket in buckets:
    bucket_df = df.filter(pl.col("_stratify_bucket") == bucket)
    n = min(per_bucket, bucket_df.height)
    sampled_parts.append(bucket_df.sample(n=n, seed=SEED))

sampled_df = pl.concat(sampled_parts)

# If we're short of target (small buckets), top up randomly from the remainder
shortfall = TARGET_GOLDEN_SIZE - sampled_df.height
if shortfall > 0:
    remaining = df.join(sampled_df.select("customer_tweet_id"), on="customer_tweet_id", how="anti")
    topup = remaining.sample(n=min(shortfall, remaining.height), seed=SEED)
    sampled_df = pl.concat([sampled_df, topup])

print(f"   Final golden set size: {sampled_df.height} rows")
print(f"   Bucket distribution:\n{sampled_df['_stratify_bucket'].value_counts()}")

print("Step 4a: Shuffling rows so buckets aren't grouped in blocks...")
sampled_df = sampled_df.sample(fraction=1.0, shuffle=True, seed=SEED)

print("Step 4: Writing golden_set_template.csv for hand-labeling...")
labeled_template = sampled_df.with_columns(
    [
        pl.lit("").alias("true_intent"),
        pl.lit("").alias("true_routing"),       # auto_handle / escalate
        pl.lit("").alias("escalation_reason"),
    ]
).select(
    [
        "customer_tweet_id",
        "customer_text",
        "apple_reply_text",
        "is_dm_redirect",
        "_stratify_bucket",  # kept as a hint while you hand-label; delete before submission if you prefer
        "true_intent",
        "true_routing",
        "escalation_reason",
    ]
)
labeled_template.write_csv(GOLDEN_OUTPUT)

print("Step 5: Saving excluded tweet_ids so the RAG index never sees eval examples...")
sampled_df.select("customer_tweet_id").write_csv(EXCLUDED_IDS_OUTPUT)

print(f"Done. Open '{GOLDEN_OUTPUT}' in Excel to hand-label true_intent / true_routing / escalation_reason.")
print(f"When building your ChromaDB index, anti-join against '{EXCLUDED_IDS_OUTPUT}' first.")