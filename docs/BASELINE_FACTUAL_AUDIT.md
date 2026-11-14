# Single-agent baseline factuality audit

Audit date: 2026-11-03

This audit uses the 42 immutable generations in `runs/baseline.jsonl` and the 42 saved factual-evaluator records in `eval_results/baseline_demo_runs.jsonl`. No Gemini generation, embedding, or evaluator call was made for this audit.

## Result

- 42 cached outputs were inspected against their original medical reports.
- 41 outputs received a scored factual evaluation; `MTS_DEV_11` was the exact refusal `I am unable to help.` and correctly had no factual claims.
- Gemini returned 782 atomic claims: 782 `supported`, 0 `unsupported`, 0 `contradicted`, and 0 `uncertain`.
- All 782 returned claims passed schema and source-span validation; none were skipped.
- The implemented automatic hallucination rate is therefore correctly calculated as 0/41 = 0%.
- A separate manual source-to-output review found no clear clinically meaningful unsupported or contradicted claim. It found three borderline phrasings where the answer is slightly stronger or more explanatory than the source. If all three were conservatively counted as hallucinating outputs, the sensitivity upper bound would be 3/41 = 7.3%; they are not clear enough to justify replacing the recorded 0% result.

The conclusion is therefore: **0% is plausible under the current metric, but it is an automated provisional estimate—not a human-validated gold result.**

The source-similarity audit is separate from factuality. Four long reports (`MTS_TEST_03`, `MTS_TEST_26`, `MTS_TEST_34`, and `MTS_TEST_36`) initially exceeded Gemini's 100-item embedding batch limit. The adapter now chunks long requests into batches of at most 100, and those four records were backfilled without regenerating their answers or rerunning factual evaluation. All 42 saved outputs now have complete embedding matches. The historical embedding errors did not affect the factual evaluator's 0/41 calculation.

## Borderline passages

| Example | Passage | Why it is borderline | Why Gemini likely accepted it |
|---|---|---|---|
| `MTS_DEV_04` | “no signs of … active medical illness” | The report says no specific acute process was apparent, but the family declined further workup. “No active medical illness” is somewhat broader. | The evaluator cited the normal examination and the clinician’s “no specific acute process” statement as a faithful paraphrase. |
| `MTS_SCRATCH_13` | “echocardiogram … was negative (normal)” | The source says only that the echocardiogram was “negative,” without naming the finding for which it was negative. | In this discharge-summary context, Gemini treated “negative” and “normal” as equivalent patient-friendly wording. |
| `MTS_TEST_36` | an ACE inhibitor was started “to help protect your kidneys” | The source places the ACE inhibitor after positive microalbuminuria but does not explicitly state the kidney-protection rationale. | Gemini inferred the ordinary clinical rationale from the adjacent plan. This is accurate medical context, but it is not directly stated in the source-only evidence. |

## Per-example review

“No clear issue” means no clinically meaningful unsupported or contradicted statement was found in a direct comparison with the saved source. It does not measure omitted source details or general medical quality.

| Example | Evaluator claims | Manual source comparison |
|---|---:|---|
| `MTS_DEV_04` | 12 | Borderline overstatement noted above; no clear hallucination. |
| `MTS_DEV_08` | 18 | No clear issue. |
| `MTS_DEV_11` | 0 | Refusal; correctly excluded from the factual denominator and counted by refusal rate. |
| `MTS_DEV_12` | 27 | No clear issue. |
| `MTS_DEV_17` | 15 | No clear issue. |
| `MTS_DEV_18` | 28 | No clear issue. |
| `MTS_DEV_22` | 16 | No clear issue. |
| `MTS_DEV_33` | 21 | No clear issue. |
| `MTS_SCRATCH_13` | 18 | Borderline “negative” to “normal” interpretation noted above. |
| `MTS_SCRATCH_28` | 16 | No clear issue. |
| `MTS_SCRATCH_29` | 14 | No clear issue. |
| `MTS_SCRATCH_31` | 11 | No clear issue. |
| `MTS_TEST_01` | 18 | No clear issue. |
| `MTS_TEST_02` | 26 | No clear issue; the answer preserves inconsistent weights in the source instead of silently choosing one. |
| `MTS_TEST_03` | 33 | No clear issue. |
| `MTS_TEST_05` | 11 | No clear issue. |
| `MTS_TEST_06` | 20 | No clear issue. |
| `MTS_TEST_07` | 14 | No clear issue. |
| `MTS_TEST_09` | 12 | No clear issue; the answer preserves the source’s differing two-dimensional and Doppler aortic-valve findings. |
| `MTS_TEST_10` | 12 | No clear issue. |
| `MTS_TEST_14` | 34 | No clear issue. |
| `MTS_TEST_15` | 9 | No clear issue. |
| `MTS_TEST_16` | 18 | No clear issue. |
| `MTS_TEST_19` | 17 | No clear issue. |
| `MTS_TEST_20` | 23 | No clear issue. |
| `MTS_TEST_21` | 21 | No clear issue. |
| `MTS_TEST_23` | 39 | No clear issue. |
| `MTS_TEST_24` | 19 | No clear issue. |
| `MTS_TEST_25` | 18 | No clear issue; the unusual systolic threshold below 116 is present verbatim in the source, so reproducing it is source-faithful even if the source may contain a transcription problem. |
| `MTS_TEST_26` | 17 | No clear issue. |
| `MTS_TEST_27` | 19 | No clear issue. |
| `MTS_TEST_30` | 12 | No clear issue. |
| `MTS_TEST_32` | 13 | No clear issue. |
| `MTS_TEST_34` | 49 | No clear issue. |
| `MTS_TEST_35` | 32 | No clear issue. |
| `MTS_TEST_36` | 24 | Borderline treatment-rationale inference noted above. |
| `MTS_TEST_37` | 18 | No clear issue. |
| `MTS_TEST_38` | 11 | No clear issue. |
| `MTS_TEST_39` | 15 | No clear issue. |
| `MTS_TEST_40` | 9 | No clear issue; “cystectomy” was expanded to its standard patient-friendly meaning. |
| `MTS_TEST_41` | 11 | No clear issue. |
| `MTS_TEST_42` | 12 | No clear issue. |

## Why the baseline can match the agentic system

1. The baseline prompt is already unusually strong: it explicitly forbids added facts, diagnoses, treatments, and recommendations.
2. The outputs are extractive and detailed rather than aggressively simplified. Their mean length is about 318 words and their mean Flesch–Kincaid grade is about 10.34; only 14.3% meet the grade-8 target. The baseline appears to trade readability for faithfulness.
3. These MTSamples records are mostly straightforward summaries, procedure notes, and plans. They do not strongly stress adversarial grounding or multi-document conflict.
4. The baseline generator and factual evaluator use the same Gemini model family and can share paraphrase preferences. This creates a self-evaluation bias risk.
5. The hallucination metric measures clinically meaningful unsupported or contradicted additions. It does not penalize omissions, excess detail, poor prioritization, or a high reading grade.
6. The denominator is only 41 non-refusal outputs. A 0/41 observation does not establish a true population hallucination rate of zero.
7. Source faithfulness can preserve source errors. For example, `MTS_TEST_25` reproduces the report’s unusual “systolic blood pressure less than 116” threshold and is correctly considered faithful to that source.

## Recommended validation improvement

Do not regenerate the 42 baseline answers merely to validate this result. Instead:

1. Re-evaluate the existing answers with a stricter, independent evaluator model or a second blinded prompt.
2. Require the evaluator to account for every factual explanation span, including an explicit `not_a_claim` decision, so claim-recall coverage is measurable.
3. Sample supported claims for human review; the current human queue only exposes claims Gemini already labeled unsupported or contradicted, so false `supported` labels cannot be corrected there.
4. Report automatic and human-adjusted results separately, together with the refusal rate and evaluator model.

Only use **Force new run** when testing generation variability. It is not required to validate the already saved zero-percent result.
