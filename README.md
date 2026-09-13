# AppleSupport AI Agent

An AI support agent for **@AppleSupport**, built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset. For each incoming customer tweet, the agent:

1. **Classifies intent** into one of 6 categories derived from the data
2. **Drafts a reply** grounded in how AppleSupport has historically resolved similar issues (RAG over ~26K real historical replies)
3. **Decides auto-handle vs. escalate** with a stated reason (rule-based, not LLM-based — see [decision_log.md](decision_log.md#4))

Full write-up, results, and failure analysis: see [REPORT.md](REPORT.md).

---

## Repo Structure

```
├── data_prep/
│   ├── prepare_data.py       # Raw twcs.csv -> cleaned customer/reply pairs
│   └── build_rag_index.py    # Builds the local ChromaDB retrieval index
├── golden_set/
│   ├── create_golden_sample.py     # Stratified sampling for the 200-row golden set
│   ├── annotation_copilot.py       # LLM-assisted labeling suggestions (co-pilot, not autopilot)
│   ├── apply_manual_overrides.py   # Documented human corrections to AI suggestions
│   ├── finalize_golden_set.py      # Produces the clean eval-ready golden set
│   └── flag_ambiguous_rows.py      # Surfaces rows needing closer human review
├── pipeline/
│   ├── full_pipeline.py       # The complete agent: classify -> retrieve -> route -> draft
│   ├── api_cache.py           # On-disk response cache (see "Reproducibility" below)
│   ├── baselines.py           # Trivial + simple baselines for comparison
│   ├── eval_harness.py        # LLM-judge scoring + human-agreement check
│   ├── intent_classifier.py   # Standalone batched intent classifier
│   ├── query_rag.py           # Retrieval quality sanity-check tool
│   └── test_groq_key.py       # Verifies API key/model access
├── data/
│   ├── chroma_db/                          # Persistent vector index (generated)
│   ├── apple_support_pairs.csv             # Cleaned historical pairs (~26K rows)
│   ├── golden_excluded_ids.csv             # Golden-set IDs excluded from RAG (leakage prevention)
│   ├── golden_set_final.csv                # The 200-row hand-reviewed golden set
│   ├── golden_set_labeling_audit_FINAL.csv # Full audit trail: AI suggestion vs. human final label
│   ├── pipeline_output.csv                 # Full pipeline output on the golden set (pre-computed)
│   ├── judge_scores.csv                    # LLM-judge quality scores (pre-computed)
│   ├── human_scores.csv                    # Human quality scores for agreement check (pre-computed)
│   └── api_cache.json                      # Cached API responses (see below)
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
python prepare_data.py       # ~1-2 min: cleans raw twcs.csv into apple_support_pairs.csv
python build_rag_index.py    # ~2-3 min: builds the local ChromaDB index (first run also downloads the embedding model, ~130MB)
```

> Raw `twcs.csv` (the Kaggle dataset) is not included in this repo due to size — download it from Kaggle and place it in `data_prep/` before running `prepare_data.py`. Alternatively, skip this step entirely: `data/apple_support_pairs.csv` and `data/chroma_db/` are already included, pre-built.

---

## Reproduce Headline Results (~5 minutes)

All commands below are run from inside the `data/` folder, since that's where the golden set, RAG index, and API cache live.

```bash
cd data

python ../pipeline/full_pipeline.py --sample 20     # classify + retrieve + route + draft, 20-tweet sample
python ../pipeline/baselines.py                     # trivial + simple baseline comparison
python ../pipeline/eval_harness.py --sample 20       # LLM-judge quality scoring
```

**Why this is fast and reliable regardless of live API conditions:** `api_cache.json` (committed to this repo) stores every API response from our full 200-row run, keyed by exact prompt. Re-running the same inputs hits the cache instead of the network — no rate-limit exposure, no variance in run time. See [decision_log.md](decision_log.md#11).

To run against the full 200-row golden set instead of a 20-row sample (also fast, since it's fully cached):
```bash
python ../pipeline/full_pipeline.py --sample 0
python ../pipeline/eval_harness.py
```

---

## Full Reported Results (already computed)

These files are the actual results reported in [REPORT.md](REPORT.md) and were generated once, in full, from a live run:

| File | Contents |
|---|---|
| `data/pipeline_output.csv` | Full pipeline output (intent, routing, reason, drafted reply) on all 200 golden-set rows |
| `data/judge_scores.csv` | LLM-judge relevance/groundedness/tone scores on all 200 replies |
| `data/human_scores.csv` | Human-scored subset (stratified sample) used for the judge-agreement check |

To regenerate any of these from scratch with live API calls (not needed for review, but available):
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
| **Full pipeline** | **~76-78%** | see `pipeline_output.csv` |

Reply quality (LLM-judge, 1-5 scale, 200 replies): relevance 4.45, groundedness 4.29, tone 4.54.

See [REPORT.md](REPORT.md) for the full breakdown, confusion matrix, failure analysis, and — importantly — why these headline numbers are more fragile than they look.

---

## Golden Set

200 hand-reviewed examples, stratified across 6 intents derived from the data (not imported from an external taxonomy). Labeled with LLM-assisted suggestions, then independently reviewed and corrected by hand — full methodology and the agreement-rate correction story in [decision_log.md](decision_log.md#3) and [REPORT.md](REPORT.md#golden-set-methodology).

---

## Key Design Decisions

The 17 non-obvious decisions behind this build — model switches, the rule-based routing choice, the caching strategy, bugs found and fixed with evidence — are all in [decision_log.md](decision_log.md).

---

## Citations / Borrowed Components

- Dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (Kaggle, thoughtvector)
- LLM inference: [Groq API](https://groq.com) (`openai/gpt-oss-120b` for classification/generation, `openai/gpt-oss-20b` for judging)
- Embeddings: `BAAI/bge-small-en-v1.5` via `sentence-transformers` (local, free, CPU)
- Vector store: [ChromaDB](https://www.trychroma.com/) (local, persistent)
- No external code was copied verbatim; all scripts were written for this project with AI coding assistance (Claude) per the assignment's rules.

---

## Known Limitations

See [REPORT.md](REPORT.md#what-id-do-next) for the full list. Briefly: single-turn only (no conversation memory), English-only, intent taxonomy has known ambiguity at the `ios_update_issues`/`out_of_scope_or_other` boundary, and reported accuracy is sensitive to golden-set sample size and labeling revisions — see the mandatory "misleading headline number" section in the report.
