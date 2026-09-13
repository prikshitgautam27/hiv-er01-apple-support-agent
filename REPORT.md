# Report: AppleSupport AI Agent

## 1. Problem Framing

**Brand chosen: AppleSupport**
- High tweet volume
- Consistent (heavily templated) support style — makes historical-pattern grounding tractable
- Diverse issue mix — forces a real taxonomy decision rather than importing one

**What "good" means for this brand:**
- Correct routing of a customer's actual problem to one of a small set of intents
- A reply that reads like something AppleSupport would plausibly send, grounded in real historical responses
- Conservative, explainable escalation whenever money, account security, safety, or an already-upset customer is involved
- A wrong troubleshooting suggestion is annoying; a wrongly auto-handled fraud complaint is a real failure — these are not equally bad

**Intents (derived from the data, not imported):**
- `battery_power_drain`
- `ios_update_issues`
- `connectivity_hardware`
- `apple_id_account`
- `app_store_billing`
- `out_of_scope_or_other` — a deliberate, honest catch-all, not a forced fit

**What we chose not to build:**
- Multi-turn conversation handling — each tweet is scored independently
- Non-English support
- Any account-level action (refunds, password resets) — the agent drafts and routes, it never *acts*
- A hosted/live demo — reproducibility from the repo was prioritized over a demo link

---

## 2. System Architecture

```
Tweet -> [1. Intent classifier, batched LLM] -> [2. RAG retrieval, local ChromaDB]
            -> [3. Rule-based escalation guardrail] -> [4. Grounded reply, LLM]
```

**Key design choice:** step 3 (auto_handle vs. escalate) is deterministic rules, not an LLM call.
- It's the highest-stakes decision in the pipeline
- Rules are auditable, consistent, and can't hallucinate a justification
- The LLM's role is classification and drafting, where pattern-matching genuinely helps

---

## 3. Golden Set Methodology

- 200 examples, stratified across the 6 intents (~33 per bucket)
- Sampled from ~26,000 cleaned historical AppleSupport pairs
- Golden-set tweet IDs excluded from the RAG index (no evaluation leakage)
- Labels produced via LLM co-pilot suggestion, then **independently human-reviewed**
- First review pass returned 100% human/AI agreement — treated as a red flag, not a good result, since it meant suggestions were being copied rather than checked
- Went back, read the model's own stated uncertainty per row, applied **16 genuine overrides** (fixing real intent disagreements + a systematic bug where the AI over-used one escalation reason for generic troubleshooting handoffs)
- **Final agreement: 96.5% intent, 97.0% routing** — a number we can defend as actually reviewed

---

## 4. Results vs. Baselines

| System | Intent Accuracy | Routing Accuracy |
|---|---|---|
| **Trivial** (majority class, always escalate, one canned reply) | 22.0% | 39.5% |
| **Simple** (keyword/regex rules, template replies, same rule-based routing as the real pipeline) | 66.0% | 64.5% |
| **Full pipeline** (LLM classify + RAG-grounded reply + rule-based routing) | 72.5–78%* | — |

*Accuracy varied across runs (72.5%, 76%, 78%, 77.5%) depending on golden-set revision and prompt version — see Section 6.

**Reply quality (LLM-judge, 1–5 scale, n=200):**
- Relevance: 4.45–4.80
- Groundedness: 4.29–4.40
- Tone: 4.54–4.60

**Judge-vs-human agreement (stratified 25-row sample):**

| Dimension | Spearman ρ | p-value | Exact match | Within 1 point |
|---|---|---|---|---|
| Relevance | 0.502 | 0.011 | 48% | 80% |
| Tone | 0.439 | 0.028 | 56% | 84% |
| Groundedness | 0.223 | 0.284 | 24% | 48% |

- Relevance and tone: statistically significant, meaningful agreement — confirmed by both correlation and exact-match/within-1 metrics moving together
- Groundedness: weak on *every* metric, not just Spearman — see Failure Mode #5
- The trivial → simple → full progression is monotonic and meaningful: keyword rules recover most easy cases; the LLM pipeline's real value-add is on the harder, ambiguous ~10-15% of traffic

---

## 5. Failure Analysis — Top 5

**1. `ios_update_issues` / `out_of_scope_or_other` confusion (biggest error cell, every run)**
- Tweets mentioning an update as background context but complaining about a different symptom get inconsistently classified
- 10–17 misclassifications across runs, consistently the single largest error cell
- *Hypothesis:* separating root cause from symptom is a genuinely hard judgment call — human labelers disagreed on several of these same tweets during golden-set review

**2. Reply contradicting its own routing decision**
- Example: "what's the best iPhone X case?" correctly routed `auto_handle`, but the drafted reply still asked the customer to DM
- Cause: retrieved historical examples happened to be DM-handoffs; reply generation wasn't yet constrained by the routing decision
- *Fix:* explicitly forbid DM requests in auto_handle prompts — caught via manual output review, not any automated metric

**3. Agent ignoring an explicit customer instruction**
- Customer tweet: *"No need to DM. Press the side button to sleep..."*
- First drafted reply: *"Can you DM us your iPhone model..."* — directly contradicting the customer
- Cause: generation pattern-matched typical AppleSupport style over actual instruction-following
- *Fix:* explicit rule respecting stated customer preferences in both routing and generation
- Clearest example in the project of an LLM optimizing for surface pattern over instruction-following

**4. Over-triggering escalation rule on benign questions**
- Rule: "escalate if majority of retrieved precedents were DM handoffs"
- Fired on a harmless case-recommendation question, since ~33% of *all* historical replies are DM redirects regardless of stakes
- A 2-out-of-3 majority was too easy to hit by chance
- *Fix:* raised threshold to require unanimous (3/3) agreement before triggering this escalation reason

**5. LLM-judge failed to catch technical hallucinations (groundedness)**
- Human review surfaced replies the judge scored as highly grounded despite real inaccuracies (e.g. outdated device-generation UI references, incorrect hardware capability claims)
- Not a fluke of one metric: groundedness scored worst on Spearman (0.223, not significant), worst on exact match (24%), worst on within-1-point (48%) — all three converge
- *Hypothesis:* groundedness for domain-specific factual claims needs either a heavier judge model or a supplementary deterministic check against retrieved precedent text, not just an LLM's general plausibility sense

---

## 6. What Is Misleading About My Headline Number

- **Accuracy is not one stable number** — read 72.5%, 76%, 78%, and 77.5% across different runs, moved by golden-set relabeling, a reverted prompt experiment, and normal run-to-run variance. Any single reported percentage implies more precision than the underlying measurement has.

- **The reverted prompt experiment is a concrete example:** added few-shot examples + explicit disambiguation rule, intended as an improvement — measurably *hurt* `ios_update_issues` recall (0.61 → 0.45) instead of helping. Reverted based on evidence, not intuition.

- **Sample size matters more than it looks:** a 30-tweet quick-demo run showed 66.7% accuracy — 10+ points below the full 200-row number, purely from sampling noise. Small `--sample` runs should never be treated as the reported result.

- **A "clean" metadata signal was wrong for most of the project:** the `is_dm_redirect` flag — used in golden-set stratification and escalation-rule design — undercounted real DM-handoffs (20% measured vs. ~33% actual) for most of the build, only caught by manually reading RAG retrieval output.

- **Near-zero correlation ≠ an unreliable judge:** first human-vs-judge check used pure random sampling and returned a Spearman correlation near zero — looked damning, but was a **ceiling effect** (both judge and human scores clustered tightly at 4-5, leaving little variance to detect). Re-sampling to include the judge's lowest-scored replies produced legitimate, significant correlation on 2 of 3 dimensions. Lesson: a bad-looking number can itself be misleading if you don't check *why* it's bad.

- **72% intent accuracy alone would not be sufficient for production autonomy** — but the escalation guardrail runs independently of intent classification (checks raw text for safety/security/billing/anger signals directly), so a misclassified intent degrades reply specificity, not safety. Production readiness would require either substantially higher intent accuracy, or treating intent as a soft signal for reply-drafting only, never a safety-relevant decision.

---

## 7. What I'd Do With One More Week

- **Larger, doubly-reviewed golden set** (400+ rows, two independent human reviewers, reported inter-annotator agreement) — 200 rows and one reviewer proves the method, not enough to fully trust the number
- **Multi-turn context** — real support threads are conversations, not isolated tweets; this is the project's biggest scope cut
- **A heavier or specialized judge model for groundedness specifically** — Failure Mode #5 shows the current judge misses technical hallucinations a human catches easily
- **A small fine-tuned local classifier** distilled from the LLM's own labeled outputs — removes the API dependency and rate-limit fragility from classification entirely
- **Pairwise/comparative LLM-judging** instead of absolute 1–5 scoring — should reduce the ceiling-effect problem in Section 6 by construction
