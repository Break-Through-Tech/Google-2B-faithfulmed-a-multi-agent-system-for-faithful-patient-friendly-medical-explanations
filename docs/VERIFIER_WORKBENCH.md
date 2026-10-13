# Verifier workbench

Launch from an activated project virtual environment:

```bash
python -m streamlit run workbench.py
```

The workbench is Verifier-only. Its active three-set manifest is [`evals/verifier_evaluation_splits_v2_balanced.json`](../evals/verifier_evaluation_splits_v2_balanced.json). The historical v1 manifests remain preserved. Each active set contains ten complete machine-answer records with distinct Question IDs, and no Question ID appears in more than one set. Each has exactly five human-gold answer-accuracy `yes` examples and five `no` examples, selected with its saved fixed seed; the manifest records every `(question_id, method_id)` pair and the sentence-relevance/evidence-relation label counts.

These are **balanced diagnostic sets**: the 50/50 answer-accuracy distribution makes it easier to inspect both classes, but does not represent the natural prevalence of `yes` and `no` in MedAESQA. Raw accuracy from these sets is therefore not a natural-population performance estimate. Only answer accuracy is balanced; the recorded sentence-relevance and evidence-relation counts are descriptive, not balance targets.

- **Prompt-development** retains previously scored examples. Inspect errors and revise prompts here.
- **Validation** compares candidate prompt versions. Once validation results influence a prompt choice, treat validation as development data rather than untouched final evidence.
- **Final-test** is unseen during development and validation. Run it only after locking the prompt, model, and settings. If its results cause a change, mark it as development use and create a new unseen final-test split for the next final evaluation.

To inspect another existing record, type its Question ID and method ID (for example, `127` and `M6`) under **Add a specific record** and click **Add and select**. The workbench validates that the exact pair exists in the local MedAESQA JSON and includes it in the next run. Clear added records to return to table-only selection.

The workbench makes at most ten Gemini requests per selected answer batch, and requests occur only after clicking **Run**. Completed raw responses are cached in `runs/verifier_workbench_cache.jsonl` by model, prompt version, and exact permitted input, so normal Streamlit reruns reuse them. Each completed run is logged locally in `runs/verifier_workbench_runs.jsonl` with the split name, selected IDs, prompt version, model, and selection seed.

The workbench uses the retained **Revised prompt (v8)**. The UI retains the current per-item review and filters; an incorrect item also shows a compact heuristic diagnostic—**Label Disagreement**, **Missing Context**, or **Reasoning Failure**—which is not a human rationale.

Each Gemini request includes only the question ID/text, method ID, complete machine answer, sentence IDs/text, and each citation ID/PMID plus its `evidence_support` excerpt. It does not include the expert-curated answer, nuggets, human labels, gold fields, scores, or annotation metadata. The revised prompt first assesses the complete answer, then sentence relevance in answer context, then evidence relation for each usable citation; answer accuracy, relevance, and evidence relation remain separate tasks. The required JSON response contains `question_id`, `method_id`, an `answer_accuracy` `yes`/`no` prediction and brief reason, plus one relevance prediction and brief reason per supplied sentence and one evidence-relation prediction plus reason per citation that has an evidence excerpt. Outer Markdown JSON fences are accepted; malformed or ID-misaligned responses are displayed but never scored as wrong.

The workbench compares three distinct human tasks after generation:

- Answer accuracy: `is_answer_accurate` `yes`/`no`.
- Sentence relevance: `required`, `borderline`, `unnecessary`, or `inappropriate`.
- Evidence relation: `supporting`, `contradicting`, `neutral`, `not relevant`, or `invalid citation`.

Citation entries without an evidence excerpt, including the dataset's `not relevant` and `invalid citation` cases in the current JSON, are skipped instead of being mapped to a negative label. The **Download review summary (JSON)** control exports separate `answer_accuracy`, `sentence_relevance`, and `evidence_relation` sections for the completed run.

## Agreement and Cohen’s kappa

For each task, the workbench scores only pairs where both the MedAESQA gold label and Gemini prediction are valid labels for that task. It reports the number of scored pairs, correct and incorrect counts, exact agreement, error rate, unweighted Cohen’s kappa (κ), and a confusion matrix. Its rows (vertical axis) are **Human gold / actual** labels; its columns (horizontal axis) are **Gemini predictions**. Each cell counts the pair at that row and column, so row `no`, column `yes` means the human gold is `no` and Gemini predicted `yes`. The table separately shows true skipped pairs (including citations with no evidence excerpt), total unscored pairs, missing gold labels, omitted predictions, invalid gold/prediction labels, and malformed responses. None of those cases are counted as correct or incorrect.

Exact agreement is the direct share of scored pairs that match. Cohen’s kappa adjusts that agreement for the agreement expected from the two observed label distributions, which is useful when labels are imbalanced and it works without declaring a single label to be “positive.” Kappa is therefore different from F1: F1 measures precision/recall for a chosen positive class (or an averaging scheme), while κ is symmetric agreement across the full label set after a chance-agreement adjustment.

The challenge documentation’s calibration target is at least 80% exact agreement and κ ≥ 0.60 for answer accuracy and evidence relation; the workbench shows this target beside those two metrics. It does not declare a pass/fail target for sentence relevance because the project documentation does not define one. When there are zero eligible pairs, κ is **Not evaluated**. A rater using a single label does not by itself make κ undefined: for example, human labels of nine `yes` and one `no` with ten Gemini `yes` predictions produce κ = 0.000. κ is **Undefined** only when expected chance agreement is one, making its denominator zero.

The saved split manifests identify examples only; their complete answer text and human annotations remain in the local MedAESQA dataset. Gold labels and expert answers are never sent to Gemini and are used only for review and scoring after a prediction is returned.

For a later MTSamples workflow, the clinical report itself becomes the evidence context. That workflow does not need external PubMed citations; it must still keep its source report separate from any labels used for scoring.
