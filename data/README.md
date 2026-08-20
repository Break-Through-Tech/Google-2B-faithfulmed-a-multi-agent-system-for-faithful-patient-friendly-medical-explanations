# Datasets

All datasets are public and either de-identified, synthetic, or expert-curated. **No PHI/PII.**
Large corpora are git-ignored — download them locally into this folder.

## Start here: MedAESQA (primary evaluation set)
- **What:** 40 health questions, each with 1 expert answer + 30 machine answers, 7,651 evidence
  excerpts, and **human accuracy / evidence-support judgments**. Expert "nuggets" = atomic facts.
- **Use:** Verifier calibration (human labels), Extractor gold (nuggets), faithfulness evaluation.
  **Evaluation only — 40 questions is too small to train on.**
- **License:** CC BY 4.0 (cite the *Scientific Data* 2025 paper).
- **Download:** OSF https://osf.io/ydbzq → save as `data/medaesqa_v1.json`.
- **Code:** https://github.com/deepaknlp/MedAESQA (see `medaesqa_eval.py`).
- `MedAESQA_methods_M1-M30.xlsx` (in this folder) documents how each of the 30 machine answers
  was generated (query, retrieval, reranking, generation, citations) — reference only.

> ⚠️ The exact JSON field names (e.g. `expert_curated_nuggets`) are taken from the paper; confirm
> them against the downloaded `medaesqa_v1.json` before wiring code to specific keys.

## Simplification corpora (the readability core)
- **PLABA** — professional ↔ plain-language pairs. https://osf.io/rnpmf/ (Attal et al.,
  *Scientific Data* 2023). **Research use — cite the paper, do not redistribute.** Gold pairs for the Simplifier.
- **MTSamples** — ~5,000 de-identified transcribed reports. https://www.mtsamples.com/ (or Kaggle
  mirror). Source documents to simplify. De-identified — handle with care, don't re-post raw records.

## Supporting data
- **MedQuAD** — ~47k consumer-health Q&A. https://github.com/abachaa/MedQuAD.
- **Synthea** — synthetic patient records (FHIR). https://synthetichealth.github.io/synthea/.
  **FHIR is deeply nested — write a flattener before feeding to the LLM. Never pass raw FHIR.**
- **MedlinePlus glossary** — lay-language definitions; the RAG knowledge base for the Simplifier.
  https://medlineplus.gov/.

## Optional benchmarks (clinical-accuracy dimension only)
- **MedQA** — USMLE-style multiple-choice. https://github.com/jind11/MedQA.
- **PubMedQA** — yes/no/maybe over abstracts. https://pubmedqa.github.io/.
- These test *clinical reasoning*, a different task than simplification/faithfulness. Optional —
  the 12-week cap is real; don't let them pull focus from the core.

## Rules of the road
1. **Freeze your gold set.** Once you curate/annotate examples, version them and stop editing.
2. **Report every headline number on the frozen set.**
3. **Never commit raw datasets or API keys** — this folder is git-ignored by design.
