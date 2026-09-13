"""
annotation_copilot.py
-----------------------
An LLM-assisted CO-PILOT for labeling the golden set -- NOT an autopilot.

IMPORTANT METHODOLOGY NOTE (put this in your decision log):
The assignment requires a "hand-labelled" golden set because it must be an
INDEPENDENT source of truth to grade your AI system against. If the same
kind of model that classifies also generates the ground truth, your
evaluation becomes circular and proves nothing.

This script only writes SUGGESTIONS to new columns:
    suggested_intent, suggested_routing, suggested_reason, model_confidence_note

You must still review every row and manually fill in the real columns:
    true_intent, true_routing, escalation_reason
(copy the suggestion if you agree, overwrite it if you don't).

This keeps the golden set legitimately "hand-labelled with AI assistance"
rather than "AI-labelled", which is the defensible version of this story.

Run: python annotation_copilot.py
Requires: pip install polars pydantic requests python-dotenv
Output: golden_set_ai_suggestions.csv
"""

import os
import time
import json
import polars as pl
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field
from typing import Literal

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise SystemExit("GROQ_API_KEY not found in .env")

INPUT_FILE = "golden_set_template.csv"
OUTPUT_FILE = "golden_set_ai_suggestions.csv"
MODEL = "openai/gpt-oss-120b"

INTENTS = [
    "battery_power_drain",
    "ios_update_issues",
    "connectivity_hardware",
    "apple_id_account",
    "app_store_billing",
    "out_of_scope_or_other",
]

SYSTEM_PROMPT = f"""You are a careful annotation assistant for a customer support dataset.
You will be shown a customer tweet to @AppleSupport and Apple's real historical reply.

Your job is ONLY to suggest labels for a human reviewer -- you are not making the
final decision. Be honest about ambiguity rather than confidently guessing.

Classify into exactly one intent from this fixed list:
{', '.join(INTENTS)}

Definitions:
- battery_power_drain: battery life, charging, overheating, dies fast
- ios_update_issues: an update/upgrade caused a problem, or asks about update availability
- connectivity_hardware: WiFi/Bluetooth/signal issues, or physical hardware problems (screen, buttons, GPS)
- apple_id_account: login, password, locked account, Apple ID issues
- app_store_billing: wrong charge, refund request, subscription/payment issue
- out_of_scope_or_other: doesn't clearly fit above, or mixes multiple issues, or is praise/spam

Then decide routing:
- "escalate" if: the reply requires private account info (DM handoff), involves money/refunds,
  account security/lockout, an angry customer needing human tone, or any safety issue.
- "auto_handle" otherwise (simple troubleshooting, general informational questions).

If escalate, give a short escalation_reason as a snake_case phrase
(e.g. requires_private_account_info, financial_refund_request, account_security_risk).
If auto_handle, escalation_reason must be an empty string.

Respond with ONLY a JSON object, no other text, in this exact shape:
{{"intent": "...", "routing": "auto_handle" or "escalate", "escalation_reason": "...", "confidence_note": "one short sentence on why this is/isn't ambiguous"}}
"""


class Suggestion(BaseModel):
    intent: Literal[
        "battery_power_drain",
        "ios_update_issues",
        "connectivity_hardware",
        "apple_id_account",
        "app_store_billing",
        "out_of_scope_or_other",
    ]
    routing: Literal["auto_handle", "escalate"]
    escalation_reason: str = Field(default="")
    confidence_note: str = ""


def get_suggestion(customer_text: str, apple_reply: str, max_retries: int = 5) -> Suggestion | None:
    user_msg = f"Customer tweet: {customer_text}\n\nApple's real reply: {apple_reply}"
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": 250,
                    "reasoning_effort": "low",
                    "temperature": 0,
                },
                timeout=30,
            )
            if response.status_code == 429:
                # Respect the server's requested wait time if it gives one,
                # otherwise back off exponentially: 5s, 10s, 20s, 40s...
                retry_after = response.headers.get("retry-after")
                wait = float(retry_after) if retry_after else (5 * (2 ** attempt))
                print(f"   [rate limit] waiting {wait:.0f}s before retrying...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"].strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            return Suggestion(**parsed)
        except (requests.RequestException, json.JSONDecodeError, ValidationError, KeyError) as e:
            print(f"   [warning] suggestion failed (attempt {attempt + 1}): {e}")
            time.sleep(3)
    return None


def main():
    print(f"Loading {INPUT_FILE}...")
    df = pl.read_csv(INPUT_FILE)

    suggested_intents, suggested_routings, suggested_reasons, confidence_notes = [], [], [], []

    total = df.height
    for i, row in enumerate(df.iter_rows(named=True)):
        print(f"Labeling row {i + 1}/{total}...", end="\r")
        suggestion = get_suggestion(row["customer_text"], row["apple_reply_text"])
        if suggestion:
            suggested_intents.append(suggestion.intent)
            suggested_routings.append(suggestion.routing)
            suggested_reasons.append(suggestion.escalation_reason)
            confidence_notes.append(suggestion.confidence_note)
        else:
            suggested_intents.append("")
            suggested_routings.append("")
            suggested_reasons.append("")
            confidence_notes.append("FAILED - label manually")
        time.sleep(2.2)  # stay under 30 requests/minute (Groq free-tier limit, per-org not per-key)

    print("\nDone. Writing suggestions...")
    df = df.with_columns(
        [
            pl.Series("suggested_intent", suggested_intents),
            pl.Series("suggested_routing", suggested_routings),
            pl.Series("suggested_reason", suggested_reasons),
            pl.Series("model_confidence_note", confidence_notes),
        ]
    )
    df.write_csv(OUTPUT_FILE)
    print(f"Wrote {OUTPUT_FILE}.")
    print("\nNEXT STEP (do not skip): open this file, review EVERY row's suggestion,")
    print("then fill in true_intent / true_routing / escalation_reason yourself --")
    print("copy the suggestion where you agree, override it where you don't.")


if __name__ == "__main__":
    main()
