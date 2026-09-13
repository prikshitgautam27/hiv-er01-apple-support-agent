"""
apply_manual_overrides.py
----------------------------
Applies MANUAL, human-decided overrides to the golden set, for the 36
rows the AI itself flagged as uncertain (via model_confidence_note).

Each override below reflects an independent human judgment call, not
an AI suggestion -- this is what makes the golden set a legitimate
independent check rather than circular AI-grades-AI validation.

Two categories of override were applied:
  1. Genuine intent/routing disagreements with the AI's suggestion
     (e.g. ambiguous "black screen" tweets reclassified from
     ios_update_issues to connectivity_hardware based on symptom
     pattern, not update-specific language)
  2. A systematic bug found in the AI's escalation_reason: it
     overused "requires_private_account_info" for cases where Apple
     was just asking to continue troubleshooting in DM (asking for
     country / iOS version / steps tried) rather than actually
     requesting account credentials or personal identity info. A new
     reason category, "requires_dm_troubleshooting", was introduced
     to correctly distinguish these.

Rows NOT listed here (20 of the 36 flagged) were reviewed and judged
correct as originally suggested -- often because the model's own
"ambiguous" flag was a false positive (e.g. saying "no ambiguous
elements", which still matched the keyword search).

Run: python apply_manual_overrides.py
Input: golden_set_labeling_audit.csv (or golden_set_ai_suggestions.csv
       if you haven't renamed it yet)
Output: golden_set_labeling_audit_FINAL.csv, golden_set_final.csv
"""

import polars as pl

INPUT_FILE = "golden_set_labeling_audit.csv"
AUDIT_OUTPUT = "golden_set_labeling_audit_FINAL.csv"
FINAL_OUTPUT = "golden_set_final.csv"

# tweet_id -> (true_intent, true_routing, escalation_reason)
# Only rows that needed a real change from the AI's suggestion are listed.
OVERRIDES = {
    671346: ("app_store_billing", "escalate", "account_security_risk"),  # fraud, not a refund ask; fixed reason formatting
    552198: ("connectivity_hardware", "auto_handle", ""),                 # black-screen symptom, no update mentioned
    588153: ("out_of_scope_or_other", "auto_handle", ""),                 # "no problem" isn't actually an issue
    924573: ("ios_update_issues", "escalate", "requires_dm_troubleshooting"),
    1101027: ("out_of_scope_or_other", "escalate", "requires_dm_troubleshooting"),  # mixed battery+service issue
    1131478: ("battery_power_drain", "escalate", "requires_dm_troubleshooting"),
    234394: ("battery_power_drain", "escalate", "requires_dm_troubleshooting"),     # was wrongly "requires_private_account_info"
    624193: ("out_of_scope_or_other", "auto_handle", ""),                 # just a wish/comment, no real issue
    310265: ("ios_update_issues", "escalate", "requires_dm_troubleshooting"),
    501503: ("connectivity_hardware", "escalate", "requires_dm_troubleshooting"),   # black screen -> hardware-leaning
    795800: ("connectivity_hardware", "auto_handle", ""),                 # bluetooth-related, not iOS update
    907432: ("out_of_scope_or_other", "escalate", "angry_customer_needs_human_tone"),  # standardized reason phrase
    87017: ("out_of_scope_or_other", "auto_handle", ""),                  # just states iOS version, no ask -- fixed inconsistent reason on auto_handle row
    716592: ("out_of_scope_or_other", "auto_handle", ""),                 # just an iOS version, no real question
    107361: ("connectivity_hardware", "auto_handle", ""),                 # root cause is broken keyboard, not login
    540929: ("connectivity_hardware", "escalate", "requires_dm_troubleshooting"),   # same black-screen-crash pattern
}

print(f"Loading {INPUT_FILE}...")
df = pl.read_csv(INPUT_FILE)

true_intent = df["true_intent"].to_list()
true_routing = df["true_routing"].to_list()
escalation_reason = df["escalation_reason"].to_list()
tweet_ids = df["customer_tweet_id"].to_list()

changed = 0
for i, tid in enumerate(tweet_ids):
    if tid in OVERRIDES:
        intent, routing, reason = OVERRIDES[tid]
        true_intent[i] = intent
        true_routing[i] = routing
        escalation_reason[i] = reason
        changed += 1

print(f"Applied {changed} manual overrides out of {len(OVERRIDES)} planned.")
if changed != len(OVERRIDES):
    print(f"WARNING: expected {len(OVERRIDES)} matches but only found {changed} -- check tweet_ids exist in the file.")

df = df.with_columns(
    [
        pl.Series("true_intent", true_intent),
        pl.Series("true_routing", true_routing),
        pl.Series("escalation_reason", escalation_reason),
    ]
)

# Recompute agreement
intent_agree = (df["true_intent"] == df["suggested_intent"]).sum()
routing_agree = (df["true_routing"] == df["suggested_routing"]).sum()
total = df.height
print(f"\nHuman-vs-AI agreement (report this number!):")
print(f"  Intent agreement:  {intent_agree}/{total} ({intent_agree/total:.1%})")
print(f"  Routing agreement: {routing_agree}/{total} ({routing_agree/total:.1%})")

df.write_csv(AUDIT_OUTPUT)
print(f"\nWrote {AUDIT_OUTPUT}")

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
print(f"Wrote {FINAL_OUTPUT} -- use this for your classifier/eval code.")
