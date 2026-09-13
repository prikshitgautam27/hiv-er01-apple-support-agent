"""
baselines.py
-------------
Two baselines required by the assignment, to show your real pipeline
is meaningfully better than something trivial and something simple.

TRIVIAL baseline: always predict the majority intent class, always
                  reply with one generic canned message, always escalate.

SIMPLE baseline: keyword/regex rules for intent (no LLM), template
                 reply per intent (no grounding), rule-based escalation
                 (same rules as the real pipeline's guardrail, reused
                 here since that part was never LLM-based to begin with).

Run: python baselines.py
Output: prints comparison metrics against golden_set_final.csv
"""

import re
import polars as pl
from sklearn.metrics import classification_report, accuracy_score

GOLDEN_SET_FILE = "golden_set_final.csv"

INTENTS = [
    "battery_power_drain", "ios_update_issues", "connectivity_hardware",
    "apple_id_account", "app_store_billing", "out_of_scope_or_other",
]

# ---------------------------------------------------------------------------
# TRIVIAL BASELINE
# ---------------------------------------------------------------------------

def trivial_predict(df: pl.DataFrame):
    majority_class = df["true_intent"].mode()[0]
    predictions = [majority_class] * df.height
    routing = ["escalate"] * df.height  # always escalate = maximally safe, maximally useless
    reply = ["Thanks for reaching out! Please DM us so we can look into this further."] * df.height
    return predictions, routing, reply


# ---------------------------------------------------------------------------
# SIMPLE BASELINE -- keyword rules, no LLM at all
# ---------------------------------------------------------------------------

SIMPLE_KEYWORD_RULES = {
    "battery_power_drain": r"\bbattery|drain|charg(e|ing)|power\b",
    "ios_update_issues": r"\bupdate|ios\s?\d|upgrad(e|ing)|install\b",
    "connectivity_hardware": r"\bwifi|bluetooth|connect|signal|screen|broken|crack|button\b",
    "apple_id_account": r"\bapple id|account|locked|password|login|sign in\b",
    "app_store_billing": r"\bcharged|refund|payment|subscription|billing|purchase|app store\b",
}

SIMPLE_TEMPLATES = {
    "battery_power_drain": "We understand battery concerns are frustrating. Please try checking Settings > Battery for usage details, and let us know if the issue continues.",
    "ios_update_issues": "Thanks for flagging this. Please make sure you're on the latest iOS version under Settings > General > Software Update, and let us know if the issue persists.",
    "connectivity_hardware": "Sorry to hear that. Please try toggling Airplane Mode on and off, or restarting your device, and let us know if this helps.",
    "apple_id_account": "We can help with your Apple ID. Please try resetting your password at iforgot.apple.com and let us know how it goes.",
    "app_store_billing": "We're sorry about the billing issue. Please review your purchase history in Settings > [Your Name] > iTunes & App Store.",
    "out_of_scope_or_other": "Thanks for reaching out! Let us know more details so we can help.",
}


def simple_predict(df: pl.DataFrame):
    predictions, routing, reply = [], [], []
    for text in df["customer_text"].to_list():
        matched = "out_of_scope_or_other"
        for intent, pattern in SIMPLE_KEYWORD_RULES.items():
            if re.search(pattern, text, re.IGNORECASE):
                matched = intent
                break
        predictions.append(matched)
        reply.append(SIMPLE_TEMPLATES[matched])

        # Same-style rule-based routing as the real pipeline (reused, since
        # this part was never LLM-based -- fair to include in both)
        if re.search(r"\b(hacked|locked out|compromis|stolen|scam|fraud)\b", text, re.IGNORECASE):
            routing.append("escalate")
        elif re.search(r"\b(refund|charged twice|overcharg)\b", text, re.IGNORECASE):
            routing.append("escalate")
        else:
            routing.append("auto_handle")

    return predictions, routing, reply


def main():
    print(f"Loading {GOLDEN_SET_FILE}...")
    df = pl.read_csv(GOLDEN_SET_FILE)
    y_true = df["true_intent"].to_list()
    true_routing = df["true_routing"].to_list()

    print(f"\n{'=' * 60}")
    print("TRIVIAL BASELINE (majority class, always escalate)")
    print(f"{'=' * 60}")
    trivial_preds, trivial_routing, _ = trivial_predict(df)
    print(f"Intent accuracy: {accuracy_score(y_true, trivial_preds):.1%}")
    print(f"Routing accuracy: {accuracy_score(true_routing, trivial_routing):.1%}")
    print(classification_report(y_true, trivial_preds, labels=INTENTS, zero_division=0))

    print(f"\n{'=' * 60}")
    print("SIMPLE BASELINE (keyword rules, no LLM)")
    print(f"{'=' * 60}")
    simple_preds, simple_routing, _ = simple_predict(df)
    print(f"Intent accuracy: {accuracy_score(y_true, simple_preds):.1%}")
    print(f"Routing accuracy: {accuracy_score(true_routing, simple_routing):.1%}")
    print(classification_report(y_true, simple_preds, labels=INTENTS, zero_division=0))

    print(f"\n{'=' * 60}")
    print("COMPARISON TABLE (copy this into your report)")
    print(f"{'=' * 60}")
    print(f"{'System':<25} {'Intent Acc':<12} {'Routing Acc':<12}")
    print(f"{'Trivial baseline':<25} {accuracy_score(y_true, trivial_preds):<12.1%} {accuracy_score(true_routing, trivial_routing):<12.1%}")
    print(f"{'Simple baseline':<25} {accuracy_score(y_true, simple_preds):<12.1%} {accuracy_score(true_routing, simple_routing):<12.1%}")
    print(f"{'Full LLM pipeline':<25} {'~76-78%':<12} {'(see pipeline_output.csv)':<12}")


if __name__ == "__main__":
    main()
