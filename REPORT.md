# Report: AppleSupport AI Agent

## 1. Problem Framing

**Brand chosen: AppleSupport.** High tweet volume, a consistent (heavily templated) support style that makes historical-pattern grounding tractable, and a genuinely diverse issue mix — enough to force a real taxonomy decision rather than importing one.

**What "good" means for this brand:** correct routing of a customer's actual problem to one of a small set of intents; a reply that reads like something AppleSupport would plausibly send, grounded in how they've actually responded to similar issues before; and — most importantly — conservative, explainable escalation whenever money, account security, safety, or an already-upset customer is involved. A wrong troubleshooting suggestion is annoying; a wrongly auto-handled fraud complaint is a real failure.

**Intents (derived from the data, not imported):** `battery_power_drain`, `ios_update_issues`, `connectivity_hardware`, `apple_id_account`, `app_store_billing`, `out_of_scope_or_other`. Arrived at by inspecting a sample of the raw threads and clustering on recurring complaint types; `out_of_scope_or_other` exists deliberately as an honest catch-all rather than forcing every tweet into a technical bucket.

**What we chose not to build:**
- Multi-turn conversation handling (each tweet is scored independently; real support threads are multi-turn, but this adds significant complexity for a scoped assignment)
- Non-English support (the dataset and prompts assume English)
- Any account-level action (refunds, password resets) — the agent drafts and routes, it never *acts*
- A hosted/live demo — reproducibility from the repo was prioritized over a demo link (see Decision Log)

---

## 2. System Architecture

```
Tweet -> [1. Intent classifier, batched LLM] -> [2. RAG retrieval, local ChromaDB]
            -> [3. Rule-based escalation guardrail] -> [4. Grounded reply, LLM]
```

**Key design choice:** step 3 (auto_handle vs. escalate) is **deterministic rules, not an LLM call.** This is the highest-stakes decision in the pipeline; rules are auditable, consistent, and can't hallucinate a justification. The LLM's role is classification and drafting, where its pattern-matching ability genuinely helps.

---

## 3. Golden Set Methodology

200 examples, stratified sampling across the 6 intents (~33 per bucket) from ~26,000 cleaned historical AppleSupport pairs, with golden-set tweet IDs excluded from the RAG index to prevent evaluation leakage.

Labels were produced with an LLM co-pilot (suggestions only) and then **independently human-reviewed**. The first review pass came back at 100% human/AI agreement — a red flag, not a good result, since it meant suggestions were being copied rather than checked. We went back, read the model's own stated uncertainty per row, and applied 16 genuine overrides (fixing both real intent disagreements and a systematic bug where the AI over-used one escalation reason for cases that were really just generic troubleshooting handoffs). Final agreement: **96.5% intent, 97.0% routing** — a number we can defend as actually reviewed, not just plausible-looking.

---

## 4. Results vs. Baselines

| System | Intent Accuracy | Routing Accuracy |
|---|---|---|
| **Trivial** (majority class, always escalate, one canned reply) | 22.0% | 39.5% |
| **Simple** (keyword/regex rules, template replies, same rule-based routing as the real pipeline) | 66.0% | 64.5% |
| **Full pipeline** (LLM classify + RAG-grounded reply + rule-based routing) | 72.5–78%* | — |

*Accuracy varied across runs (72.5%, 76%, 78%, 77.5%) depending on golden-set revision and prompt version — see Section 6.

**Reply quality** (LLM-judge, 1–5 scale, n=200): relevance 4.45–4.80, groundedness 4.29–4.40, tone 4.54–4.60 depending on run.

**Judge-vs-human agreement** (stratified 25-row human sample, to avoid a ceiling-effect artifact from pure random sampling — see Section 6):

| Dimension | Spearman ρ | p-value | Exact match | Within 1 point |
|---|---|---|---|---|
| Relevance | 0.502 | 0.011 | 48% | 80% |
| Tone | 0.439 | 0.028 | 56% | 84% |
| Groundedness | 0.223 | 0.284 | 24% | 48% |

Relevance and tone show statistically significant, meaningful agreement — confirmed by both the correlation and the more robust exact-match/within-1 metrics moving together. Groundedness is weak on *every* metric, not just Spearman — see Failure Mode #5 for why.

The trivial→simple→full progression is monotonic and meaningful: keyword rules alone recover most of the easy cases, and the LLM pipeline's real value-add is on the harder, more ambiguous ~10-15% of traffic.

---

## 5. Failure Analysis — Top 5

**1. `ios_update_issues` / `out_of_scope_or_other` confusion (biggest error cell, every run).**
Tweets that mention an iOS update as background context but complain about a different symptom (battery, apps freezing) get inconsistently classified — sometimes correctly tied to the update, sometimes dumped into the catch-all. Confusion matrix showed 10–17 misclassifications here across runs, consistently the single largest error cell. *Hypothesis: the taxonomy asks the model to separate root cause from symptom, which is a genuinely hard judgment call — human labelers disagreed on several of these same tweets during golden-set review.*

**2. Reply contradicting its own routing decision.**
Example: a tweet asking "what's the best iPhone X case?" was correctly routed `auto_handle`, but the drafted reply still asked the customer to DM — because the retrieved historical examples happened to be DM-handoffs, and reply generation initially wasn't constrained by the routing decision. *Fixed by explicitly forbidding DM requests in auto_handle prompts; caught via manual output review, not by any automated metric.*

**3. Agent ignoring an explicit customer instruction.**
Customer tweet: *"No need to DM. Press the side button to sleep..."* — the first version of the drafted reply responded with *"Can you DM us your iPhone model..."*, directly contradicting what the customer had just said, because generation pattern-matched to typical AppleSupport style regardless of context. *Fixed with an explicit rule respecting stated customer preferences in both routing and generation. This is the clearest example in the whole project of an LLM optimizing for surface pattern over actual instruction-following.*

**4. Over-triggering escalation rule on benign questions.**
The rule "escalate if the majority of retrieved historical precedents were DM handoffs" fired on a harmless "what case should I buy" question, since ~33% of *all* historical replies are DM redirects regardless of stakes — a 2-out-of-3 majority was too easy to hit by chance. *Fixed by raising the threshold to require unanimous (3/3) agreement among retrieved precedents before triggering this specific escalation reason.*

**5. LLM-judge failed to catch technical hallucinations (groundedness).**
Manual human review during the stratified agreement check surfaced replies the judge scored as highly grounded despite containing real inaccuracies (e.g., referencing outdated device-generation UI navigation, or asserting a hardware capability not applicable to the device in question). This wasn't a fluke of one metric: groundedness scored worst on Spearman correlation (0.223, not significant), worst on exact match (24%), and worst on within-1-point agreement (48%) — all three converge on the same conclusion. The lighter judge model (`gpt-oss-20b`, chosen for a separate quota bucket) could evaluate tone and relevance well but lacked the depth to verify device-specific technical claims. *Hypothesis: groundedness for domain-specific factual claims needs either a heavier judge model or a supplementary deterministic check (e.g., validating claims against retrieved precedent text directly), not just an LLM's general sense of plausibility.*

---

## 6. What Is Misleading About My Headline Number

**Accuracy is not one stable number.** Across the project it read 72.5%, 76%, 78%, and 77.5% across different runs — moved by golden-set relabeling (a 16-row correction), a reverted prompt experiment (few-shot + explicit disambiguation, which *measurably hurt* `ios_update_issues` recall from 0.61 to 0.45 despite being intended as an improvement), and normal run-to-run variance. **Any single reported percentage implies more precision than the underlying measurement actually has.**

**Sample size matters more than it looks.** A 30-tweet quick-demo run showed 66.7% accuracy — 10+ points below the full 200-row number — purely from sampling noise. Anyone reproducing this with a small `--sample` value should not treat that run's number as the reported result.

**A "clean" metadata signal was wrong for most of the project.** The `is_dm_redirect` flag — used both in golden-set stratification and in escalation-rule design — was undercounting real DM-handoffs (20% measured vs. ~33% actual) for most of the build, only caught by manually reading RAG retrieval output rather than trusting the regex.

**Near-zero correlation ≠ an unreliable judge.** Our first human-vs-judge agreement check used pure random sampling and returned a Spearman correlation near zero — which looks damning, but was actually a **ceiling effect**: both judge and human scores were clustered tightly at 4–5 (most replies genuinely were good), and rank correlation is mathematically unstable with almost no variance to detect. Re-sampling to deliberately include the judge's lowest-scored replies produced a legitimate, statistically significant correlation on 2 of 3 dimensions. The lesson: a bad-looking correlation number can itself be misleading if you don't check *why* it's bad.

---

## 7. What I'd Do With One More Week

- **Larger, doubly-reviewed golden set** (400+ rows, two independent human reviewers, inter-annotator agreement reported) — 200 rows and one reviewer is enough to prove the method, not enough to fully trust the number.
- **Multi-turn context**, since real support threads are conversations, not isolated tweets — the single-turn assumption is this project's biggest scope cut.
- **A heavier or specialized judge model for groundedness specifically**, since Failure Mode #5 shows the current judge misses technical hallucinations that a human catches easily.
- **A small fine-tuned local classifier** (distilled from the LLM's own labeled outputs) to remove the API dependency and rate-limit fragility entirely from the classification step.
- **Pairwise/comparative LLM-judging** instead of absolute 1–5 scoring, which should reduce the ceiling-effect problem observed in Section 6 by construction.
