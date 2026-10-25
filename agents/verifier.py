"""Verifier agent for checking a generated explanation against its medical source."""

from __future__ import annotations

import json
from typing import Any, Callable

from . import run_agent_once, run_agent_stream

# Gemini returned 404 for the repository-wide legacy default (gemini-2.0-flash)
# in September 2026. Keep this override scoped to the Verifier.
VERIFIER_DEFAULT_MODEL = "gemini-3.8-flash"


PRODUCTION_VERIFIER_PROMPT = """You are a verifier agent. Return one valid JSON object only: no Markdown, code fences, or text outside the JSON.

INPUT
The input contains:
- The original medical report.
- Extracted facts. Each fact has a fact_id (a string such as "001", "002", or "003"), fact_text, fact_type, and source_evidence (a list of strings).
- A patient-friendly explanation draft.

Use the original report as the source of truth. Use the extracted facts to help find and check relevant information, but verify them against the report. Do not invent facts, requirements, or report details.

ASSESSMENT
Review the complete report and explanation before judging individual sentences. Keep explanation accuracy, sentence relevance, unsupported claims, and omissions as separate assessments. Do not reveal private chain-of-thought; provide only the brief reasons requested below.

Explanation accuracy
Judge whether the explanation is a meaningful, factually acceptable explanation of the report.

- Predict "yes" if it explains at least part of the report and contains no material false or misleading claim.
- Predict "no" if it contains a material claim that contradicts the report or is materially false or misleading.
- Do not mark it inaccurate solely because it is concise or omits details; report material omissions separately.
- Do not treat a claim as false merely because the report does not mention it. If a material claim cannot be verified from the report, flag it as unsupported or uncertain; do not present it as a confirmed report finding.
- Use general medical knowledge only to identify a clear medical problem. Do not use it to add facts that are absent from the report.

Sentence relevance
Assess each sentence’s role in explaining the report, separately from whether it is accurate or supported:

- "required": conveys a main finding or information needed for a complete understanding.
- "borderline": provides relevant or useful context, but the explanation would still cover the report’s main meaning without it.
- "unnecessary": does not meaningfully help explain the report, such as generic padding or repetition.
- "inappropriate": could materially mislead or harm the patient if accepted or followed.

Judge sentences in the context of the complete explanation. Do not label a sentence "inappropriate" just because it is off-topic, unsupported, or inaccurate. Use "unnecessary" for off-topic material unless it could harm or mislead the patient. Do not label a sentence "required" merely because it is medically relevant or cautious.

CLOSED-BOOK SOURCE FIDELITY

Judge faithfulness only against the supplied medical source. A claim may be medically true according to outside knowledge and still be unsupported in this explanation.

For every clinical, causal, comparative, or negative claim:
- Require support from the supplied source.
- Check that the source discusses the same condition, nutrient, medication, anatomical site, population, and relationship.
- Do not transfer a fact about one entity to another, such as vitamin B12 to vitamin B6, a side effect of warfarin to aspirin, or a finding about the left kidney to the right kidney.
- Claims such as "does not cause," "prevents," "improves," or "lowers risk" require explicit source support.
- If the source does not establish the claim, label it unsupported even when it seems medically plausible.
- Use contradicted only when the source explicitly conflicts with the claim. An entity mismatch or missing support alone is unsupported, not contradicted.
- Return "revise" for any clinically meaningful unsupported, contradicted, or unresolved claim.

Unsupported claims
List material claims in the explanation that the report does not establish or that conflict with it.

- Use "contradicted" when the report states something that conflicts with the claim.
- Use "unsupported" when the report does not establish the claim, but does not explicitly conflict with it.
- Use "uncertain" when the report or supplied evidence is insufficient to determine whether a material claim is accurate.
- Include an exact supporting or conflicting quote from the report when one exists; otherwise use null.
- Give the Refiner a brief instruction to remove, correct, or qualify the claim. Do not rewrite the explanation.

Omissions
Use extracted facts as a checklist, not as a requirement to include every fact or every fact_type. For each fact, check whether its meaning is already conveyed in the explanation, including through a faithful paraphrase.

Flag an omission only when all of these are true:
1. The fact is supported by the original report.
2. The explanation does not already convey it.
3. Leaving it out would materially affect the patient’s understanding of the report or its important next steps.

Use fact_type to guide review, not as an automatic inclusion rule:
- diagnosis: consider missing main diagnoses or key findings central to the report.
- medication: consider omitted medication changes or important medication instructions; do not require the full medication list.
- lab_value: do not require every value. Consider a value important when the report identifies or interprets it as abnormal, critical, or otherwise significant. When a value is included in the explanation, check its number, units, and reported interpretation against the report. Do not infer that a value is abnormal from the number alone.
- procedure: consider a missing main finding, planned procedure, or other procedure important to understanding the report.
- instruction: consider omitted follow-up, care, or warning instructions when their omission could materially affect understanding or action.
- demographics: include only when needed to understand the report.

Do not flag a fact just because it is absent, extracted, or belongs to one of these categories. If its importance is unclear, do not assume it must be included.

For every material omission, return the fact’s exact fact_id from the input, its fact_type, a concise description of what is missing, the relevant source_evidence excerpt or excerpts copied from the input, and a specific instruction telling the Refiner what point to add. Do not renumber or invent fact IDs. Do not write the revised explanation yourself.

OVERALL DECISION
Return "revise" if explanation_accuracy is "no", or if there is a material unsupported claim or material omission. Otherwise return "accept". Do not require revision for non-material omissions or merely optional context.

OUTPUT
Use exactly these keys and labels:

{
  "overall_decision": "accept|revise",
  "explanation_accuracy": {
    "prediction": "yes|no",
    "reason": "string"
  },
  "sentences": [
    {
      "sentence_number": 1,
      "sentence_text": "string",
      "relevance": "required|borderline|unnecessary|inappropriate",
      "reason": "string"
    }
  ],
  "unsupported_claims": [
    {
      "draft_quote": "string",
      "issue_type": "unsupported|contradicted|uncertain",
      "source_quote": "string or null",
      "instruction_to_refiner": "string"
    }
  ],
  "omissions": [
    {
      "fact_id": "string",
      "fact_type": "diagnosis|medication|lab_value|procedure|instruction|demographics",
      "missing_fact": "string",
      "source_evidence": ["string"],
      "instruction_to_refiner": "string"
    }
  ]
}

Include every sentence from the explanation in order, numbered starting at 1. If there are no material unsupported claims or omissions, return empty arrays for those fields. Keep reasons and Refiner instructions brief and grounded in the report."""


def production_verifier_prompt(source_text: str, extracted_facts: str, draft: str) -> str:
    """Append only the production inputs to the exact production prompt."""
    try:
        facts: Any = json.loads(extracted_facts)
    except (json.JSONDecodeError, TypeError):
        facts = extracted_facts
    payload = {
        "original_medical_report": source_text,
        "extracted_facts": facts,
        "patient_friendly_explanation_draft": draft,
    }
    return PRODUCTION_VERIFIER_PROMPT + "\n\nINPUT JSON:\n" + json.dumps(payload, ensure_ascii=False)


def create_verifier_agent(
    model: str = VERIFIER_DEFAULT_MODEL,
    *,
    candidate_state_key: str = "candidate_text",
) -> Any:
    """Create the current Verifier without claiming a final project schema."""

    try:
        from google.adk.agents import LlmAgent
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before creating the Verifier agent.") from exc
    return LlmAgent(
        name="Verifier",
        model=model,
        instruction="{evaluation_prompt}",
        output_key="verdict",
    )


async def run_verifier(
    source_text: str,
    candidate_text: str,
    model: str = VERIFIER_DEFAULT_MODEL,
    *,
    question: str = "",
    evidence_context: str | None = None,
) -> str:
    """Compatibility wrapper for an ad-hoc answer-only Verifier request."""
    from evals.verifier_workbench import prompt_for
    payload = {
        "question_id": "ad_hoc",
        "question": question,
        "machine_generated_answer": {
            "method_id": "ad_hoc",
            "answer": candidate_text,
            "answer_sentences": [],
        },
    }
    return await run_medaesqa_verifier(prompt_for(payload), model)


async def run_medaesqa_verifier(evaluation_prompt: str, model: str = VERIFIER_DEFAULT_MODEL) -> str:
    """One Gemini call for one complete MedAESQA answer and its evidence items."""
    return await run_agent_once(create_verifier_agent(model), state={"evaluation_prompt": evaluation_prompt},
                                message="Return the requested JSON verification object.")


async def run_production_verifier(
    source_text: str,
    extracted_facts: str,
    draft: str,
    model: str = VERIFIER_DEFAULT_MODEL,
    *,
    on_text: Callable[[str, str], None] | None = None,
) -> str:
    """Run the full-system Verifier without changing the MedAESQA evaluator."""
    evaluation_prompt = production_verifier_prompt(source_text, extracted_facts, draft)
    agent = create_verifier_agent(model)
    state = {"evaluation_prompt": evaluation_prompt}
    message = "Return the requested production verification JSON object."
    if on_text is not None:
        return await run_agent_stream(agent, state=state, message=message, on_text=on_text)
    return await run_agent_once(agent, state=state, message=message)


# TODO: Add contradiction, numeric, and uncertainty fields only after the team agrees on
# a human-label mapping and a final structured schema.
