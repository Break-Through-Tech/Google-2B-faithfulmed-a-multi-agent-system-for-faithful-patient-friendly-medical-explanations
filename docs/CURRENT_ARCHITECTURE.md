# FaithfulMed current architecture

FaithfulMed has five Google ADK generation agents, a Python production orchestrator, two runtime demos, and offline evaluation tools.

## Current tree

```text
agents/
├── __init__.py             # ADK execution helpers
├── extractor.py            # structured facts and lookup terms
├── medical_lookup.py       # Python UMLS lookup
├── simplifier.py           # patient-friendly draft
├── readability.py          # readability rewrite
├── verifier.py             # MedAESQA and production verification
├── refiner.py              # targeted corrections
└── production_pipeline.py  # observable bounded pipeline
evals/
├── system_demo.py          # datasets, cache, factual evaluation
├── readability_eval.py     # shared readability metrics
├── verifier_workbench.py   # MedAESQA workbench helpers
└── leaderboard*.py         # model comparison and reports
tests/
└── test_*.py               # offline unit tests
demo/
├── server.py               # FastAPI and saved-run replay
└── web/                    # React frontend
data/
└── validate_annotations.py # annotation validator
docs/
└── CURRENT_ARCHITECTURE.md
```

Dataset files, annotations, manifests, data documentation, `.env.example`, and the intentional baseline prompt in `prompts/baseline_v1.txt` remain in place.

## What each Python file does

- `agents/production_pipeline.py` is the single end-to-end generation path used by both demos.
- `evals/system_demo.py` runs the pipeline, independent factual evaluator, metrics, and append-only cache.
- `agentic_system_demo.py` is the Streamlit evaluation interface.
- `demo/server.py` and `demo/web/` provide the hosted FastAPI/React demo.
- `workbench.py` provides the individual Baseline, Verifier, Refiner, and Readability workbenches.
- `data/validate_annotations.py` is a separate read-only annotation validator.

## Reference definitions (UMLS)

`agents/medical_lookup.py` adds a Python-only `UMLSLookup` stage to the production pipeline, between the Extractor and the Simplifier:

1. The existing Extractor call returns two separate lists: report-grounded `patient_facts` and source-grounded `lookup_terms`. It does not generate definitions. Code keeps a lookup term only if it appears word-for-word in a cited patient fact or in exact `source_evidence`, allowing general medical documents to request definitions without inventing patient facts.
2. Every term, including medications, is looked up in UMLS. Lookups are plain code; no model writes or edits a definition. DrugBank is not part of the pipeline.
3. Every validated term is looked up; there is no silent whole-report term limit. Terms may be grounded through patient `fact_ids` or exact `source_evidence`, so educational documents do not need invented patient facts. When UMLS returns several CUIs, Python ranks and filters them using the Extractor category, nearby source context, and UMLS semantic types so drugs, substances, procedures, abbreviations, and anatomical concepts do not resolve to context-incompatible candidates. Concise MedlinePlus definitions are preferred when available. If a compatible candidate lacks an allowed definition, lookup continues through the ranked candidates. The result (`faithfulmed_reference_definitions_v4`) is stored in `trace["results"]["reference_definitions"]`, separate from the Extractor's facts. Each entry has the term, its grounding, the definition, `source` (UMLS), `source_id` (CUI), semantic types, provenance, and `allowed_use: general_term_explanation_only`. Unresolved terms record why, including `not_found`, `ambiguous`, `context_mismatch`, `no_definition`, `api_error`, or `not_configured`.
4. The Simplifier receives the definitions as its `{glossary}`. The production Verifier and independent factual evaluator receive them as a separate `reference_definitions` key. A general term explanation may cite UMLS, but every patient-specific claim still requires report evidence.

Any lookup failure is recorded in the trace and the pipeline continues without definitions. There is no Term Finder call. With no UMLS source configured, the stage is skipped without network access.

Setup (optional):

```bash
# UMLS: put a UTS API key in .env as UMLS_API_KEY=...
# Check one term (UMLS uses the network)
python3 -m agents.medical_lookup lookup metoprolol --category medication
```

UMLS definitions are accepted from MedlinePlus, MeSH, NCI, CSP, and HPO. Concise MedlinePlus definitions are preferred for patient-facing language; a truncated MedlinePlus topic page falls behind a complete concise definition. Ambiguous or context-incompatible abbreviations, semantic types, and anatomical matches are left unresolved instead of guessed. Each glossary line records its fact IDs or source-only grounding.

Smoke test for the lookup handoffs on a synthetic note (see `evals/lookup_smoke.py`):

```bash
python3 -m evals.lookup_smoke --mode mock       # offline; also run by the test suite
python3 -m evals.lookup_smoke --mode umls-live  # real UMLS, mocked Gemini stages
python3 -m evals.lookup_smoke --mode live       # real UMLS and Gemini (uses API quota)
python3 -m evals.lookup_smoke --mode live --model gemini-3.1-flash-lite  # pick the model for every stage
```

The report includes `accepted` and `final_explanation` so step 4 can be reviewed by hand.

### Recorded live runs

The entries below describe the historical v3/v4 TermFinder implementation and
are retained as an audit trail. Pipeline v5 removed that separate model call;
the Extractor now supplies `lookup_terms` in its existing response.

**2026-11-13, `gemini-3.1-flash-lite`, live UMLS** (`python3 -m evals.lookup_smoke --mode live --model gemini-3.1-flash-lite`)

- Steps 1, 2, 3a, 3b: **passed**. The historical stage order was Extractor, TermLookup, Simplifier, Verifier, Readability; the current order places Readability before the final Verifier loop. UMLS returned C0020538 ("Hypertensive disease", MeSH definition, not truncated) linked to fact F-001, and both the Simplifier glossary and the Verifier's `reference_definitions` received it.
- Step 4: **manual faithfulness finding**. The source says "The patient has a history of hypertension." The accepted final output said "You have hypertension, which is the medical term for high blood pressure." Dropping "history of" changes the claim from a past condition to a current one, and the Verifier did not flag it. The final output also still contained the Simplifier's `PATIENT-FRIENDLY VERSION:` / `REVIEW FLAGS:` wrapper.
- Follow-up: the Simplifier and Verifier prompts now name time and status qualifiers ("history of", "prior", "suspected", "possible", "resolved", "ruled out"), and the pipeline now strips the wrapper (see below). Expected meaning on rerun: "You have a history of high blood pressure, also called hypertension. Please continue your current treatment plan."

**2026-11-13 rerun after the fixes, same model and command**

- First attempt: the TermFinder's Gemini call returned `503 UNAVAILABLE` (model under high demand), so no UMLS lookup ran and steps 2, 3a, 3b failed for that reason only. The pipeline recorded the failure and still completed without definitions, as designed. Not counted as a UMLS result.
- Second attempt: steps 1, 2, 3a, 3b **passed** (UMLS C0020538 linked to F-001 and handed to both the Simplifier and the Verifier). Step 4 **passed on manual review**. Final output: "You have a history of high blood pressure. This is medically known as hypertension. Please continue to follow your current treatment plan as directed." It keeps "history of", covers both facts, uses the UMLS definition only to explain the term, and has no wrapper headers. "As directed" is a minor addition that does not change the meaning.
- Medication lookup, live UMLS (`python3 -m agents.medical_lookup lookup metoprolol --category medication`): **found** C0025859 ("metoprolol", MeSH, not truncated): "A selective adrenergic beta-1 blocking agent that is commonly used to treat ANGINA PECTORIS; HYPERTENSION; and CARDIAC ARRHYTHMIAS." Medications resolve through UMLS now that DrugBank is out of the pipeline.
- Not yet covered live: a medication term through the full pipeline, and real (non-synthetic) notes.

### Simplifier output boundary

The Simplifier returns two sections, `PATIENT-FRIENDLY VERSION:` and `REVIEW FLAGS:`. `split_simplifier_output()` in `agents/production_pipeline.py` separates them right after the Simplifier stage. Only the patient text goes to Readability and then the Verifier/Refiner loop; the exact final text must be accepted after all readability edits. The raw output stays in `trace["results"]["simplifier"]` and the flags in `trace["results"]["simplifier_review_flags"]`. Before this, the wrapper could reach `final_explanation`, the demo display, readability metrics, and the factual evaluator's spans (10 of 32 saved runs in `eval_results/system_demo_runs.jsonl` contain it; those saved records are unchanged).

## Model leaderboard

Six configurations: {Gemini `gemini-3.8-flash`, Gemma `gemma-4-31b-it`, gpt-oss `groq/openai/gpt-oss-120b`} × {single-model baseline, full agentic system}. In a full-system configuration the one model powers every generation agent (Extractor, Simplifier, Verifier, Refiner, Readability); UMLS lookup is Python/HTTP, not an agent call. Models, prices, openness, and sources live in `evals/leaderboard_models.json`.

**The automated faithfulness judge is Gemini** (`gemini-3.8-flash`, existing factual-evaluator prompt) for all six configurations. Gemini is also a contestant, so a blind human review of an equal-sized random sample per configuration is supported; human judgments and judge scores are kept and reported separately.

- `evals/leaderboard.py` (offline): manifests, result schema, aggregation, Markdown/CSV report, blind review sampling, labelled historical runs.
- `evals/leaderboard_runner.py` (model calls): resumable runs with bounded retries, per-call token usage, and generation latency that includes retry waits and excludes the judge.
- Results: `eval_results/leaderboard/{manifests,attempts.jsonl,evaluations.jsonl,reviews,reports}`. Review keys (`review_keys/`) are gitignored until judging is finished.

```bash
python3 -m evals.leaderboard_runner status --manifest eval_results/leaderboard/manifests/smoke_dev.json
python3 -m evals.leaderboard_runner run    --manifest eval_results/leaderboard/manifests/smoke_dev.json   # API quota
python3 -m evals.leaderboard report        --manifest eval_results/leaderboard/manifests/smoke_dev.json
python3 -m evals.leaderboard report --historical   # earlier runs, labelled with their original settings
python3 -m evals.leaderboard review-sample --manifest eval_results/leaderboard/manifests/dev_pilot.json --per-config 5 --seed 20261110
```

The frozen `test_final` manifest runs only with `--final`, after the setup is fixed. Every attempt records its setup fingerprint (prompts, pipeline version, revision limit, lookup settings, package versions, model); results from different setups are never merged.

## Agent, pipeline, test, and evaluation

- An **agent** is one ADK `LlmAgent` that performs one job.
- The **production pipeline** in `agents/production_pipeline.py` connects agents, performs UMLS lookup, bounds the Verifier/Refiner loop, and records every stage.
- A **unit test** checks deterministic Python behavior with small synthetic inputs and no model call.
- An **agent evaluation** compares saved model predictions with independent human labels. Verifier metrics can be recalculated without calling Gemini again.

The production flow is Extractor, optional Python UMLS lookup, Simplifier, Readability, and the bounded Verifier/Refiner loop. The independent factual evaluator runs only after the production pipeline returns an accepted final explanation.

## Run one agent independently

Install dependencies and set `GOOGLE_API_KEY` only when you intend to make a model call. This example runs only the Extractor:

```bash
# CALLS GEMINI AND MAY USE API QUOTA
python3 - <<'PY'
import asyncio
from agents.extractor import run_extractor

result = asyncio.run(run_extractor("Scratch development medical source goes here."))
print(result)
PY
```

Use only scratch or development material while building prompts. The same pattern applies to `run_simplifier()`, `run_refiner()`, and `run_readability()` with their documented arguments. The full pipeline is not involved.

## Run the Verifier independently

Start `python -m streamlit run workbench.py`, choose **Verifier**, and select scratch or development records. The workbench keeps human labels out of the model prompt and caches raw responses by model, prompt version, and exact input.

## Evaluate the Verifier independently

Generation and scoring are separate:

1. Generate and cache raw predictions through the Verifier workbench.
2. Parse the saved output with the matching evaluator.
3. Recalculate metrics later from saved predictions and independent human labels. This step is offline and makes no Gemini call.

MedAESQA has two distinct Verifier-relevant levels:

- **Answer accuracy:** `is_answer_accurate` is mapped only as `yes` → acceptable (`faithful=True`) and `no` → not acceptable (`faithful=False`). This is the dataset's answer-level judgment, not a label for the narrower assertion “contains an unsupported claim.”
- **Citation support:** `evidence_relation` is mapped only as `supporting` → support and `contradicting` → contradiction. A citation verifier gets the question, one answer sentence, and that citation's `evidence_support` snippet. `neutral`, `not relevant`, missing evidence, and invalid/missing citations are unavailable—not negative examples.

`answer_sentence_relevance` (`required`, `borderline`, `unnecessary`, `inappropriate`) measures answer relevance/completeness and is not a factual-support target. Expert answers are not string-match targets. Nuggets are not citation-support labels. The M1–M30 spreadsheet describes systems, not example labels.

These mappings follow the [MedAESQA paper](https://doi.org/10.1038/s41597-025-05233-z) and its [upstream evaluator](https://github.com/deepaknlp/MedAESQA/blob/main/src/medaesqa_eval.py): the upstream evaluator reports answer accuracy, citation support rate, and citation contradiction rate as distinct measures. It does not define a binary factual-support metric for `neutral` or `not relevant`.

```python
from evals.verifier_eval import load_medaesqa_examples, load_predictions, score_predictions

gold = load_medaesqa_examples("data/medaesqa_v1.json", level="answer")
metrics = score_predictions(gold, load_predictions("predictions.jsonl"))
print(metrics)
```

Score answer and citation examples in separate calls. Malformed model output and unavailable labels are excluded and reported separately. A zero denominator produces `None`, not a fake zero. IDs preserve question, method, sentence, and citation where applicable, so false positives and negatives remain traceable.

## Shared evaluation code

Each evaluator has one plainly named module in `evals/`. Shared readability calculations live in `evals/readability_eval.py`; the end-to-end cache and factual metrics live in `evals/system_demo.py`. Offline tests exercise these paths without calling Gemini.

## Data and test-set safety

Use scratch and development examples while designing prompts or choosing models. Do not use the frozen test split for iteration. Looking at test results while tuning makes the final measurement unreliable.

The MedAESQA JSON is absent in this checkout, so no project metrics were generated. Once it is available locally, `load_medaesqa_examples()` supports offline scoring of saved predictions. Do not tune prompts on a frozen test set; use development or scratch examples only.

## Commands

Offline and safe:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q agents evals tests notebooks
python3 data/validate_annotations.py
```

Creates ADK objects but does not itself send a model request:

```bash
python3 -m notebooks.adk_pipeline
```

The notebook command requires ADK to be installed. Any example that calls an `agents.run_*` function calls Gemini and may use API quota. Never commit `.env` or a populated key; ADK reads credentials from environment variables.

## Intentionally incomplete

- Teammate-owned agents retain starter instructions and explicit TODOs; they are not final implementations.
- The complete pipeline has not been run or medically validated.
- There is no production pipeline module, deployment code, ablation system, or generalized runner.
- There are no teammate evaluation files yet.
- Citation scoring intentionally leaves neutral, not-relevant, missing-evidence, and invalid-citation cases unavailable because MedAESQA does not define them as supported or contradicted.
- Answer and citation prediction generation require a model call; loading saved predictions and scoring them does not.
