# Decision Log

Non-obvious decisions made while building this pipeline, and why. In roughly chronological order.

---

**1. Chose AppleSupport as the brand.**
High tweet volume, a consistent (templated) support style that makes historical-pattern grounding tractable, and a genuinely diverse issue mix (battery, iOS updates, connectivity, account, billing) — enough to force a real taxonomy decision rather than a trivial one.

**2. Used an LLM co-pilot to *suggest* golden-set labels, but required full human review before anything counted as ground truth.**
A golden set labelled by the same kind of model it's meant to evaluate is circular validation, not independent proof. The AI only ever wrote to `suggested_*` columns; `true_*` columns were a separate, human-decided pass.

**3. Treated a 100% human/AI agreement rate as a red flag, not a good result.**
First labeling pass came back with 200/200 agreement — a sign the AI's suggestions were being copied, not reviewed. Went back, read the model's own uncertainty notes (`model_confidence_note`), and applied 16 genuine overrides, landing at a defensible 96.5%/97% agreement instead of a suspicious 100%.

**4. Escalation routing is decided by deterministic rules, not the LLM.**
The auto_handle/escalate decision is the highest-stakes output in the pipeline. Rules are auditable, consistent, and can't hallucinate a justification — appropriate here even though it means the LLM's judgment isn't used for this one decision.

**5. Golden-set tweet IDs are explicitly excluded from the RAG index (anti-join on tweet_id before indexing).**
Prevents the agent from ever retrieving the answer to a question it's being evaluated on — leakage would silently inflate every downstream reply-quality metric.

**6. Deduplicated near-identical templated replies before building the RAG index.**
Apple's real replies are heavily templated; without dedup, retrieval would look artificially strong by always returning one of a handful of boilerplate answers rather than genuinely relevant precedent.

**7. Refined the `is_dm_redirect` detection regex mid-project after inspecting RAG retrieval output, rather than trusting the first version.**
Manually reading retrieved examples revealed the original regex undercounted DM-handoffs (20% measured vs. ~33% actual). Fixed the heuristic but deliberately did *not* redo golden-set labels over it, since those were independently human-reviewed and unaffected by this metadata flag.

**8. Switched models mid-project (`llama-3.3-70b-versatile` → `openai/gpt-oss-120b`) after discovering the former had moved to Groq's Enterprise-only tier.**
An external platform change discovered via a live 404, not anticipated in the original plan — documented rather than hidden.

**9. Batched intent classification (10 tweets per API call) instead of one call per tweet.**
Necessary to fit within Groq's free-tier rate limits and the assignment's 15-minute reproduction budget; ~10x fewer requests for the same classification work.

**10. Tried few-shot examples and an explicit disambiguation rule in the classifier prompt — then reverted both after measuring they hurt more than helped.**
Hypothesized these would fix a known confusion (`ios_update_issues` vs `out_of_scope_or_other`). Measured result: accuracy dropped (78%→76%) and `ios_update_issues` recall specifically got worse (0.61→0.45). Reverted based on evidence, not intuition — prompt complexity isn't free.

**11. Built a persistent on-disk cache of API responses, keyed by exact prompt hash.**
Live API rate limits are unpredictable (observed run times from under a minute to 40+ minutes on the same 200 rows). Committing the cache file to the repo makes the "reproduce in 15 minutes" claim reliable on review day regardless of Groq's real-time conditions, instead of hoping for good luck.

**12. Used a local, free, CPU-only embedding model (`bge-small-en-v1.5`) for RAG rather than an API-based embedding service.**
Removes a second rate-limit dependency from the pipeline and keeps retrieval instant and free.

**13. Reply drafting is explicitly constrained by the routing decision, not left free to contradict it.**
Found a real bug where a reply asked the customer to DM even though it had been routed `auto_handle`. Fixed by passing the routing decision into the reply-drafting prompt as a hard constraint.

**14. Added an explicit rule to respect a customer's stated preference (e.g. "no need to DM") in both routing and reply generation.**
Found a case where the agent asked a customer to DM immediately after they'd explicitly said not to — the model was pattern-matching historical style over the actual conversation content. Fixed at both the routing and generation layers.

**15. Used a separate, lighter model (`gpt-oss-20b`) for LLM-judge scoring rather than the same model used for classification/generation.**
Keeps judge scoring on its own quota bucket so it doesn't compete with the main pipeline's rate limits, and is enough model for a bounded rubric-scoring task.

**16. Diagnosed and corrected a ceiling effect in the judge/human agreement check rather than reporting a misleadingly bad correlation at face value.**
Initial random-sampled human-vs-judge comparison produced a near-zero Spearman correlation. Investigation showed both judge and human scores were tightly clustered at 4-5 (most replies were genuinely good), which makes rank correlation mathematically unstable regardless of true agreement. Fixed by stratifying the human sample to include some of the judge's lowest-scored replies, and added exact-match/within-1-point agreement as supplementary metrics that remain meaningful under score clustering.

**17. Reused the same rule-based escalation logic in the "simple" baseline as in the real pipeline, rather than giving the baseline no escalation logic at all.**
Escalation was never LLM-based in either system, so excluding it from the baseline would have made the comparison artificially favor the full pipeline for the wrong reason.
