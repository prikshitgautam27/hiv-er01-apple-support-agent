"""
test_groq_key.py
------------------
Sanity check that GROQ_API_KEY in your .env is valid before you build
the intent classifier / RAG pipeline on top of it.

Run: python test_groq_key.py
Requires: pip install python-dotenv requests
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current directory

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise SystemExit(
        "GROQ_API_KEY not found. Make sure .env is in the same folder "
        "you're running this script from, and contains:\n"
        "GROQ_API_KEY=your_key_here"
    )

print("Found key, testing a live call to Groq...")

response = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 100,       # gpt-oss models spend some tokens on hidden reasoning
        "reasoning_effort": "low",  # keeps reasoning short so more budget reaches the actual answer
    },
    timeout=15,
)

if response.status_code == 200:
    reply = response.json()["choices"][0]["message"]["content"]
    print(f"Success! Model replied: {reply!r}")
else:
    print(f"Failed ({response.status_code}): {response.text}")