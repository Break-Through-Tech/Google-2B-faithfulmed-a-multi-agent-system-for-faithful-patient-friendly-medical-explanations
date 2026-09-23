# FaithfulMed current architecture

FaithfulMed currently has five individual Google ADK agents, one notebook-based integration point, and one small Verifier evaluation module. The complete medical workflow is experimental and is not yet a finished system.

## Current tree

```text
agents/
├── __init__.py          # one-shot ADK helper
├── extractor.py         # Extractor agent
├── simplifier.py        # Simplifier agent
├── verifier.py          # Verifier agent
├── refiner.py           # Refiner agent
└── readability.py       # Readability agent
evals/
├── __init__.py
└── verifier_eval.py     # offline Verifier scoring
  #add your evals for your agents here
tests/
└── test_verifier_eval.py
  #add your own tests for your agents here
notebooks/
├── adk_pipeline.py      # current integration point
└── eval_harness.py      # provided starter evaluation script
data/
└── validate_annotations.py
docs/
└── CURRENT_ARCHITECTURE.md
```

Dataset files, annotations, manifests, data documentation, `.env.example`, and the intentional baseline prompt in `prompts/baseline_v1.txt` remain in place.

## What each Python file does

- `agents/__init__.py` contains `DEFAULT_MODEL` and `run_agent_once()`. The helper creates one isolated ADK session and returns final text.
- `agents/extractor.py` owns `create_extractor_agent()` and `run_extractor()`. Its final schema is a teammate TODO.
- `agents/simplifier.py` owns `create_simplifier_agent()` and `run_simplifier()`. Glossary retrieval and its final output contract are TODOs.
- `agents/verifier.py` owns `create_verifier_agent()` and `run_verifier()`. It currently requests the four fields from the starter notebook: `faithful`, `unsupported_claims`, `omissions`, and `reading_level_ok`.
- `agents/refiner.py` owns `create_refiner_agent()` and `run_refiner()`. The final revision contract remains a TODO.
- `agents/readability.py` owns `create_readability_agent()` and `run_readability()`. Its owner must decide whether the final result is a score, revised text, or both.
- `evals/__init__.py` exposes the small Verifier evaluation types and scoring function.
- `evals/verifier_eval.py` validates Verifier output, saves and loads JSONL predictions, and calculates unsupported-claim TP/FP/TN/FN, precision, recall, and F1 against independent human labels.
- `tests/test_verifier_eval.py` tests scoring with synthetic data only. It never imports ADK or calls Gemini.
- `notebooks/adk_pipeline.py` imports the five agent builders and assembles the current sequence plus the bounded Verifier/Refiner loop.
- `notebooks/eval_harness.py` is provided starter material. It offers readability scores, a target-grade check, a basic loader for the missing full MedAESQA file, and accuracy/Cohen's kappa helpers.
- `data/validate_annotations.py` is the existing read-only annotation validator and is separate from agent evaluation.

## Agent, pipeline, test, and evaluation

- An **agent** is one ADK `LlmAgent` that performs one job.
- The **ADK pipeline** connects agents and passes state between them. It currently lives only in `notebooks/adk_pipeline.py`.
- A **unit test** checks deterministic Python behavior with small synthetic inputs and no model call.
- An **agent evaluation** compares saved model predictions with independent human labels. Verifier metrics can be recalculated without calling Gemini again.

## Why integration stays in the notebook

The team is still developing the individual agents. Keeping integration in `notebooks/adk_pipeline.py` makes the stage order and intermediate ADK state easy to inspect while prompts and outputs are changing. After the workflow is stable, reusable orchestration may move into a normal production module. That module does not exist yet.

The notebook imports the five `create_*_agent()` functions. `build_pipeline()` creates each agent, connects Verifier and Refiner in a loop capped at two passes, then returns the sequential ADK agent. Building these objects does not call Gemini. Running the returned agent through an ADK runner does.

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

```bash
# CALLS GEMINI AND MAY USE API QUOTA
python3 - <<'PY'
import asyncio
from agents.verifier import run_verifier

raw_prediction = asyncio.run(run_verifier(
    source_text="Scratch development medical source.",
    candidate_text="Scratch generated explanation.",
))
print(raw_prediction)
PY
```

This returns raw response text. Pass it to `evals.verifier_eval.parse_prediction()` before saving or scoring it. The current output shape is provisional. Contradiction, numeric-error, and uncertainty-error fields have not been added because there is no agreed human-label mapping.

## Evaluate the Verifier independently

Generation and scoring are separate:

1. Use `generate_predictions()` with `agents.verifier.run_verifier`, or call the agent yourself.
2. Use `save_predictions()` to keep raw predictions in JSONL.
3. Later, use `load_predictions()` and `score_predictions()` with independent human labels. This step is offline and makes no Gemini call.

The binary positive class is: **the human label says the generated claim or explanation is unsupported**.

```python
from evals.verifier_eval import LabeledVerifierExample, load_predictions, score_predictions

gold = [LabeledVerifierExample(
    example_id="scratch-1",
    source_text="source",
    candidate_text="candidate",
    gold_unsupported=True,
)]
metrics = score_predictions(gold, load_predictions("predictions.jsonl"))
print(metrics)
```

Malformed model output and missing human labels are excluded and reported separately. A zero denominator produces `None`, not a fake zero. False-positive and false-negative IDs are retained for review.

## Shared evaluation code

There is no generalized evaluation framework. `evals/verifier_eval.py` contains only what current Verifier work needs. No Extractor, Simplifier, Refiner, Readability, end-to-end, ablation, or system-comparison evaluator exists.

When a teammate is ready to evaluate an agent, they should inspect the real labels available for that task, then create one plainly named evaluator and one offline test file. They should not copy Verifier label meanings or create placeholder scores. Those teammate evaluation files do not exist yet because their data contracts and metrics have not been agreed.

## Starter evaluation notebook

`notebooks/eval_harness.py` currently provides Flesch-Kincaid grade, SMOG, difficult-word fraction, word count, an eighth-grade target check, a simple `data/medaesqa_v1.json` loader, accuracy, and Cohen's kappa.

It is not executable in the current local environment until `textstat` and scikit-learn are installed. The full MedAESQA JSON is also absent. None of its functions were copied into `verifier_eval.py`; the reusable Verifier evaluator uses standard-library-only binary metrics so offline tests stay small.

## Data and test-set safety

Use scratch and development examples while designing prompts or choosing models. Do not use the frozen test split for iteration. Looking at test results while tuning makes the final measurement unreliable.

The repository contains `data/MedAESQA_methods_M1-M30.xlsx`, which documents answer-generation methods. It does not contain the questions, answers, evidence excerpts, stable IDs, or human accuracy/evidence-support judgments required for real Verifier evaluation. Real MedAESQA scoring is not currently possible, and no label mapping has been invented.

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
- The Verifier schema is provisional and deterministic evaluation currently covers unsupported-claim detection only.
- Real MedAESQA evaluation cannot run until the full labeled dataset and documented field mapping are available.
