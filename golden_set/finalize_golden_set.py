"""
finalize_golden_set.py
------------------------
Takes your fully hand-reviewed golden_set_ai_suggestions.csv and produces:

  1. golden_set_final.csv        -- clean columns only, this is what your
                                     classifier/eval harness will load
  2. golden_set_labeling_audit.csv -- full file with suggested_* columns
                                     kept, for your report/decision log as
                                     proof of the AI-assisted labeling process
  3. Prints your human-vs-AI agreement rate on intent and routing --
     genuinely useful evidence for your report ("I agreed with the AI
     co-pilot's suggestion on X% of intent labels, Y% of routing labels")

Run: python finalize_golden_set.py
"""

import polars as pl

INPUT_FILE = "golden_set_labeling_audit.csv"  # your manually-reviewed file, NOT the raw AI output
FINAL_OUTPUT = "golden_set_final.csv"
AUDIT_OUTPUT = "golden_set_labeling_audit_verified.csv"  # separate name -- never overwrite your edited source file

print(f"Loading {INPUT_FILE}...")
df = pl.read_csv(INPUT_FILE)

# Sanity check: no blanks in the required columns
missing_intent = df.filter(pl.col("true_intent").is_null() | (pl.col("true_intent") == "")).height
missing_routing = df.filter(pl.col("true_routing").is_null() | (pl.col("true_routing") == "")).height
if missing_intent or missing_routing:
    print(f"WARNING: {missing_intent} rows missing true_intent, {missing_routing} rows missing true_routing.")
    print("Fix these before finalizing -- the eval harness needs every row labeled.")

# Compute agreement rate between your final answer and the AI's suggestion
intent_agree = (df["true_intent"] == df["suggested_intent"]).sum()
routing_agree = (df["true_routing"] == df["suggested_routing"]).sum()
total = df.height

print(f"\nHuman-vs-AI agreement (report this number!):")
print(f"  Intent agreement:  {intent_agree}/{total} ({intent_agree/total:.1%})")
print(f"  Routing agreement: {routing_agree}/{total} ({routing_agree/total:.1%})")

# 1. Clean file for the eval harness
final_df = df.select(
    [
        "customer_tweet_id",
        "customer_text",
        "apple_reply_text",
        "is_dm_redirect",
        "true_intent",
        "true_routing",
        "escalation_reason",
    ]
)
final_df.write_csv(FINAL_OUTPUT)
print(f"\nWrote clean eval file: {FINAL_OUTPUT}")

# 2. Full audit file (just renaming what you already have, for clarity)
df.write_csv(AUDIT_OUTPUT)
print(f"Wrote audit trail file: {AUDIT_OUTPUT}")

print("\nDone. Use golden_set_final.csv for your classifier/eval code.")
print("Keep golden_set_labeling_audit.csv for your report's methodology section.")