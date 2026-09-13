# AppleSupport AI Agent

An AI support agent for **@AppleSupport**, built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset.

For each incoming customer tweet, the agent:
- Classifies intent into one of 6 categories derived from the data
- Retrieves grounding from ~26K real historical AppleSupport replies (local RAG)
- Decides auto-handle vs. escalate using deterministic rules (not an LLM)
- Drafts a reply, constrained by that routing decision

Full write-up: [REPORT.md](REPORT.md) · Full decision history: [decision_log.md](decision_log.md)

---

## Architecture

```
                    [ Incoming Customer Tweet ]
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   1. Intent Classifier   │   Batched (10/call), Groq gpt-oss-120b
                 └─────────────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   2. RAG Retrieval       │   Local ChromaDB, top-3 historical matches
                 └─────────────────────────┘   Golden-set IDs excluded (no leakage)
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ 3. Deterministic Router  │   Rule-based, not LLM — auditable,
                 └─────────────────────────┘   no hallucinated justification
                        │           │
                   [escalate]   [auto_handle]
                        │           │
                        ▼           ▼
                 ┌─────────────────────────┐
                 │   4. Reply Draft-Gen     │   Grounded in step 2's retrieval,
                 └─────────────────────────┘   hard-constrained by step 3's decision
                              │
                              ▼
                    [ Intent + Routing + Reply ]
```

**Why retrieval happens before routing, not after:** the router uses retrieval as a signal (e.g. "were similar past tweets historically handled via DM?"), so every tweet is retrieved first regardless of how it's ultimately routed.

**Why routing is rule-based, not LLM-based:** it's the highest-stakes decision in the pipeline. Rules are auditable and consistent; an LLM here could hallucinate a plausible-sounding but wrong justification. See [decision_log.md #4](decision_log.md).

---

## Repo Structure

```
├── data_prep/
│   ├── prepare_data.py       # Raw twcs.csv -> cleaned customer/reply pairs
│   └── build_rag_index.py    # Builds the local ChromaDB retrieval index
├── golden_set/
│   ├── create_golden_sample.py     # Stratified sampling for the 200-row golden set
│   ├── annotation_copilot.py       # LLM-assisted label suggestions (co-pilot, not autopilot)
│   ├── apply_manual_overrides.py   # Documented human corrections to AI suggestions
│   ├── finalize_golden_set.py      # Produces the clean, eval-ready golden set
│   └── flag_ambiguous_rows.py      # Surfaces rows needing closer human review
├── pipeline/
│   ├── full_pipeline.py       # classify -> retrieve -> route -> draft
│   ├── api_cache.py           # On-disk response cache (see "Reproducibility")
│   ├── baselines.py           # Trivial + simple baselines
│   ├── eval_harness.py        # LLM-judge scoring + human-agreement check
│   ├── intent_classifier.py   # Standalone batched intent classifier
│   ├── query_rag.py           # Retrieval quality sanity-check tool
│   └── test_groq_key.py       # Verifies API key/model access
├── data/
│   ├── chroma_db/                          # Persistent vector index (generated)
│   ├── apple_support_pairs.csv             # Cleaned historical pairs (~26K rows)
│   ├── golden_excluded_ids.csv             # Golden-set IDs excluded from RAG
│   ├── golden_set_final.csv                # The 200-row hand-reviewed golden set
│   ├── golden_set_labeling_audit_FINAL.csv # Full audit: AI suggestion vs. human label
│   ├── pipeline_output.csv                 # Pre-computed full pipeline output
│   ├── judge_scores.csv                    # Pre-computed LLM-judge scores
│   ├── human_scores.csv                    # Pre-computed human scores (agreement check)
│   └── api_cache.json                      # Cached API responses
├── decision_log.md
├── REPORT.md
├── requirements.txt
└── .env.example
```

---

## Setup (one-time, not part of the timed reproduction)

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your GROQ_API_KEY

cd data_prep
python prepare_data.py       # ~1-2 min: cleans raw twcs.csv
python build_rag_index.py    # ~2-3 min: builds ChromaDB (first run downloads ~130MB embedding model)
```

- Raw `twcs.csv` (Kaggle) is not included due to size — download it and place it in `data_prep/` before running `prepare_data.py`
- Or skip this step entirely: `data/apple_support_pairs.csv` and `data/chroma_db/` are already pre-built and included

---

## Reproduce Headline Results (~5 minutes)

Run from inside `data/` — that's where the golden set, RAG index, and API cache live.

```bash
cd data
python ../pipeline/full_pipeline.py --sample 20
python ../pipeline/baselines.py
python ../pipeline/eval_harness.py --sample 20
```

**Why this is fast and reliable regardless of live API conditions:**
- `api_cache.json` (committed to this repo) stores every response from our full 200-row run, keyed by exact prompt
- Re-running the same inputs hits the cache, not the network — no rate-limit exposure, no run-time variance
- See [decision_log.md #11](decision_log.md)

To run the full 200-row golden set instead (also fast, since it's cached):
```bash
python ../pipeline/full_pipeline.py --sample 0
python ../pipeline/eval_harness.py
```

---

## Full Reported Results (already computed)

| File | Contents |
|---|---|
| `data/pipeline_output.csv` | Full pipeline output on all 200 golden-set rows |
| `data/judge_scores.csv` | LLM-judge scores on all 200 replies |
| `data/human_scores.csv` | Human-scored subset used for the agreement check |

To regenerate from scratch with live API calls (not needed for review):
```bash
python ../pipeline/full_pipeline.py --sample 0
python ../pipeline/eval_harness.py
python ../pipeline/eval_harness.py --human-sample 25   # interactive, manual
python ../pipeline/eval_harness.py --correlate
```

---

## Headline Numbers

| System | Intent Accuracy | Routing Accuracy |
|---|---|---|
| Trivial baseline (majority class, always escalate) | 22.0% | 39.5% |
| Simple baseline (keyword rules, no LLM) | 66.0% | 64.5% |
| **Full pipeline** | **72.5%** | see `pipeline_output.csv` |

- LLM-judge scores (1-5): relevance 4.45, groundedness 4.29, tone 4.54
- Judge-vs-human agreement (stratified sample, n=25):
  - Relevance: Spearman ρ=0.502 (p=0.011), 80% within 1 point
  - Tone: Spearman ρ=0.439 (p=0.028), 84% within 1 point
  - Groundedness: Spearman ρ=0.223 (not significant), 48% within 1 point — see [REPORT.md](REPORT.md) for why

Full breakdown, confusion matrix, failure analysis, and why these numbers are more fragile than they look: [REPORT.md](REPORT.md).

---

## Golden Set

- 200 hand-reviewed examples, stratified across 6 intents derived from the data (not an imported taxonomy)
- Labeled via LLM-assisted suggestion, then independently human-reviewed and corrected
- First review pass returned a suspicious 100% human/AI agreement — treated as a red flag, not a result. Re-reviewed and applied 16 genuine overrides, landing at a defensible 96.5%/97% agreement
- Full audit trail: [decision_log.md #3](decision_log.md) and `data/golden_set_labeling_audit_FINAL.csv`

---

## Key Design Decisions

17 non-obvious decisions — model switches, the rule-based routing choice, the caching strategy, bugs found and fixed with evidence — in [decision_log.md](decision_log.md).

---

## Citations / Components

- Dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (Kaggle, thoughtvector)
- LLM inference: [Groq API](https://groq.com) — `openai/gpt-oss-120b` (classification/generation), `openai/gpt-oss-20b` (judging)
- Embeddings: `BAAI/bge-small-en-v1.5` via `sentence-transformers` (local, free, CPU)
- Vector store: [ChromaDB](https://www.trychroma.com/) (local, persistent)
- All code written for this project with AI coding assistance (Claude), per assignment rules — no external code copied verbatim

---

## Known Limitations

- Single-turn only — no conversation memory across a thread
- English-only
- Known ambiguity at the `ios_update_issues` / `out_of_scope_or_other` boundary (see failure analysis)
- Reported accuracy is sensitive to golden-set sample size and labeling revisions — see the mandatory "misleading headline number" section in [REPORT.md](REPORT.md#6-what-is-misleading-about-my-headline-number)
