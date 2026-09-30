# Verifier workbench

Launch from an activated project virtual environment:

```bash
python -m streamlit run workbench.py
```

The workbench is Verifier-only. It selects one to five `(question_id, method_id)` records with a deterministic greedy rule: each next record covers the largest number of human-label categories not already present. It displays the IDs before a run and makes at most one Gemini request per selected answer. Requests occur only after clicking **Run**. Completed raw responses are cached in `runs/verifier_workbench_cache.jsonl` by model, prompt version, and exact permitted input, so normal Streamlit reruns reuse them.

Each Gemini request includes only the question ID/text, method ID, complete machine answer, sentence IDs/text, and each citation ID/PMID plus its `evidence_support` excerpt. It does not include the expert-curated answer, nuggets, or any human annotation. The required JSON response contains `question_id`, `method_id`, an `answer_accuracy` `yes`/`no` prediction and brief reason, plus one relevance prediction and brief reason per supplied sentence and one evidence-relation prediction plus reason per citation that has an evidence excerpt. Outer Markdown JSON fences are accepted; malformed or ID-misaligned responses are displayed but never scored as wrong.

The workbench compares three distinct human tasks after generation:

- Answer accuracy: `is_answer_accurate` `yes`/`no`.
- Sentence relevance: `required`, `borderline`, `unnecessary`, or `inappropriate`.
- Evidence relation: `supporting`, `contradicting`, `neutral`, `not relevant`, or `invalid citation`.

Citation entries without an evidence excerpt, including the dataset's `not relevant` and `invalid citation` cases in the current JSON, are skipped instead of being mapped to a negative label. A five-answer exploratory sample may not cover every label. These results are not final research or frozen-test-set results.

For a later MTSamples workflow, the clinical report itself becomes the evidence context. That workflow does not need external PubMed citations; it must still keep its source report separate from any labels used for scoring.
