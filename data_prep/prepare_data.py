"""
prepare_data.py
----------------
Builds the AppleSupport customer<->reply pairs dataset from the raw
Kaggle twcs.csv, with fixes for:
  1. Mojibake / encoding corruption (Iï, â€™, etc.)
  2. Duplicate / near-duplicate templated Apple replies (which would
     otherwise dominate the RAG index and make retrieval look
     artificially good)
  3. A lightweight "dm_redirect" flag so we're honest that a large
     fraction of first-responses are hand-offs to DM, not resolutions
  4. Thread depth, so downstream code can tell "one-shot reply" apart
     from "multi-turn troubleshooting"

Run: python prepare_data.py
Output: apple_support_pairs.csv
"""

import re
import polars as pl
from ftfy import fix_text

RAW_FILE = "twcs.csv"
OUTPUT_FILE = "apple_support_pairs.csv"

# Phrases that indicate Apple is just redirecting to DM rather than
# actually resolving the issue in-thread. Keep this list visible/editable —
# it's a documented, inspectable heuristic, not a black box.
DM_REDIRECT_PATTERNS = [
    r"\bsend (us|you) a (private message|dm|direct message)\b",
    r"\bplease dm\b",
    r"\bmeet (us|up) in (a )?dm\b",
    r"\bdm and we'll\b",
    r"\bwe'll go from there\b",
    r"\bjoin us in a dm\b",
    r"\bdm us\b",
    r"\bdm us at\b",
    r"\bcheck your dm\b",
    r"\breceived your dm\b",
    r"\bover dm\b",
    r"\bvia dm\b",
    r"\bin your dm\b",
    r"\breach out (to us )?(via|over|through) dm\b",
    r"\b(contact|reach out to) our .* team\b",  # e.g. "contact our Account Security team"
]
DM_REDIRECT_RE = re.compile("|".join(DM_REDIRECT_PATTERNS), re.IGNORECASE)


def clean_text(s: str) -> str:
    """Repair encoding artifacts (mojibake) from the raw Twitter export."""
    if s is None:
        return s
    return fix_text(s)


print("Step 1: Reading twcs.csv lazily (utf8-lossy to avoid hard crashes on bad bytes)...")
ctx = pl.scan_csv(RAW_FILE, infer_schema_length=10000, encoding="utf8-lossy")

print("Step 2: Filtering for Apple Support interactions...")
apple_filter = ctx.filter(
    (pl.col("author_id") == "AppleSupport")
    | (pl.col("text").str.contains("(?i)@AppleSupport"))
).collect()

print("Step 3: Reconstructing customer -> first-reply pairs...")
inbound_tweets = apple_filter.filter(pl.col("inbound") == True).with_columns(
    pl.col("tweet_id").cast(pl.Int64)
)
apple_replies = apple_filter.filter(pl.col("author_id") == "AppleSupport").with_columns(
    pl.col("in_response_to_tweet_id").cast(pl.Int64)
)

structured_threads = inbound_tweets.join(
    apple_replies,
    left_on="tweet_id",
    right_on="in_response_to_tweet_id",
    how="inner",
).select(
    [
        pl.col("tweet_id").alias("customer_tweet_id"),
        pl.col("text").alias("customer_text"),
        pl.col("tweet_id_right").alias("apple_tweet_id"),
        pl.col("text_right").alias("apple_reply_text"),
    ]
)

print("Step 4: Repairing mojibake / broken encoding in text fields...")
structured_threads = structured_threads.with_columns(
    [
        pl.col("customer_text").map_elements(clean_text, return_dtype=pl.Utf8),
        pl.col("apple_reply_text").map_elements(clean_text, return_dtype=pl.Utf8),
    ]
)

print("Step 5: Flagging dm_redirect-only replies (honesty, not a resolution)...")
structured_threads = structured_threads.with_columns(
    pl.col("apple_reply_text")
    .map_elements(lambda t: bool(DM_REDIRECT_RE.search(t or "")), return_dtype=pl.Boolean)
    .alias("is_dm_redirect")
)

print("Step 6: Computing thread depth (how many turns this customer had with Apple)...")
thread_depth = (
    inbound_tweets.group_by("tweet_id")
    .agg(pl.len().alias("_dummy"))  # placeholder; real depth computed below
)
# Real depth: count total inbound tweets from the same author_id chain.
depth_lookup = (
    apple_filter.filter(pl.col("inbound") == True)
    .group_by("author_id")
    .agg(pl.len().alias("thread_depth"))
)
structured_threads = structured_threads.join(
    inbound_tweets.select(["tweet_id", "author_id"]),
    left_on="customer_tweet_id",
    right_on="tweet_id",
    how="left",
).join(
    depth_lookup, on="author_id", how="left"
).drop("author_id")

print("Step 7: Deduplicating near-identical templated Apple replies...")
before = structured_threads.height
# Normalize whitespace/case for dedup comparison only (keep original text in output)
structured_threads = structured_threads.with_columns(
    pl.col("apple_reply_text")
    .str.to_lowercase()
    .str.replace_all(r"\s+", " ")
    .str.replace_all(r"https?://\S+", "<link>")  # links are per-conversation, ignore for dedup
    .alias("_dedup_key")
)
structured_threads = structured_threads.unique(subset=["_dedup_key"], keep="first").drop("_dedup_key")
after = structured_threads.height
print(f"   Removed {before - after} near-duplicate templated replies ({before} -> {after})")

structured_threads.write_csv(OUTPUT_FILE)
print(f"Finished! Dataset created: '{OUTPUT_FILE}' with {structured_threads.height} valid QA pairs.")
print(f"   dm_redirect share: {structured_threads['is_dm_redirect'].mean():.1%}")