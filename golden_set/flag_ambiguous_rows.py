"""
flag_ambiguous_rows.py
------------------------
Scans model_confidence_note for language suggesting the AI wasn't
confident, and writes just those rows to a small file for focused
manual review. This is the practical way to catch a suspiciously
high agreement rate without re-reading all 200 rows by hand.

Run: python flag_ambiguous_rows.py
Input: golden_set_labeling_audit.csv
Output: rows_to_review.csv (a short list -- review THESE by hand)
"""

import polars as pl

INPUT_FILE = "golden_set_labeling_audit.csv"
OUTPUT_FILE = "rows_to_review.csv"

UNCERTAINTY_MARKERS = [
    "ambiguous", "could be", "could also", "unclear", "not clear",
    "doesn't clearly fit", "does not clearly fit", "either", "uncertain",
    "hard to tell", "borderline", "might be", "slightly ambiguous",
    "not entirely", "somewhat unclear",
]

df = pl.read_csv(INPUT_FILE)

pattern = "|".join(UNCERTAINTY_MARKERS)
flagged = df.filter(
    pl.col("model_confidence_note").str.to_lowercase().str.contains(pattern)
)

flagged.select(
    [
        "customer_tweet_id",
        "customer_text",
        "apple_reply_text",
        "true_intent",
        "true_routing",
        "escalation_reason",
        "suggested_intent",
        "suggested_routing",
        "suggested_reason",
        "model_confidence_note",
    ]
).write_csv(OUTPUT_FILE)

print(f"Total rows: {df.height}")
print(f"Flagged as uncertain by the model itself: {flagged.height}")
print(f"Wrote {OUTPUT_FILE} -- review and correct ONLY these rows by hand.")
print("(If this number is very small, e.g. under 15, also spot-check ~15-20")
print(" random non-flagged rows manually so your review isn't 100% AI-selected too.)")
