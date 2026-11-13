"""Observable full-system FaithfulMed pipeline used by the Streamlit demo."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from . import run_agent_stream
from .extractor import ExtractorOutput, create_extractor_agent
from .medical_lookup import (
    MedicalLookup, build_reference_definitions, create_term_finder_agent,
    empty_reference_definitions, reference_definitions_for_prompt, render_glossary,
)
from .readability import create_readability_agent
from .refiner import create_refiner_agent
from .simplifier import create_simplifier_agent
from .verifier import VERIFIER_DEFAULT_MODEL, run_production_verifier

MAX_REVISION_CYCLES = 2
TraceCallback = Callable[[dict[str, Any]], None]
AgentExecutor = Callable[[Any, dict[str, Any], str, Callable[[str, str], None]], Awaitable[str]]
# (source_text, extracted_facts, draft, model, on_text, reference_definitions_json)
VerifierRunner = Callable[[str, str, str, str, Callable[[str, str], None], str], Awaitable[str]]


@dataclass(frozen=True)
class PipelineRunResult:
    trace: dict[str, Any]
    final_explanation: str | None
    accepted: bool


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit(callback: TraceCallback | None, event: dict[str, Any]) -> None:
    if callback:
        callback(event)


def _strip_json_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, re.S)
    return match.group(1).strip() if match else text.strip()


_SIMPLIFIER_TEXT_HEADER = re.compile(r"PATIENT-FRIENDLY VERSION\s*:?[*#]*", re.I)
_SIMPLIFIER_FLAGS_HEADER = re.compile(r"\n[ \t]*[*#]*[ \t]*REVIEW FLAGS\s*:?[*#]*", re.I)


def split_simplifier_output(text: str) -> tuple[str, str | None]:
    """Split the Simplifier's two-section output into (patient text, review flags).

    Only the patient text goes on to the Verifier, Refiner, Readability, and the
    final explanation; review flags are reviewer notes, not patient text. Output
    without the expected headers is returned whole.
    """
    flags = _SIMPLIFIER_FLAGS_HEADER.search(text)
    body = text[: flags.start()] if flags else text
    header = _SIMPLIFIER_TEXT_HEADER.search(body)
    if header and not body[: header.start()].strip(" \t\n*#"):
        body = body[header.end():]
    review_flags = text[flags.end():].strip() if flags else None
    return body.strip(), review_flags or None


def parse_production_verdict(raw: str) -> dict[str, Any]:
    """Parse the production decision fields required by pipeline control flow."""
    data = json.loads(_strip_json_fence(raw))
    if not isinstance(data, dict) or data.get("overall_decision") not in {"accept", "revise"}:
        raise ValueError("Verifier response has no valid overall_decision.")
    accuracy = data.get("explanation_accuracy")
    if not isinstance(accuracy, dict) or accuracy.get("prediction") not in {"yes", "no"}:
        raise ValueError("Verifier response has no valid explanation_accuracy prediction.")
    for field in ("sentences", "unsupported_claims", "omissions"):
        if not isinstance(data.get(field), list):
            raise ValueError(f"Verifier response field {field!r} must be a list.")
    return data


def refiner_feedback(verdict: dict[str, Any]) -> dict[str, Any]:
    """Adapt the production verdict to the current Refiner's existing contract."""
    unsupported = []
    for issue in verdict.get("unsupported_claims", []):
        if isinstance(issue, dict):
            unsupported.append(issue.get("instruction_to_refiner") or issue.get("draft_quote") or json.dumps(issue))
    omissions = []
    for issue in verdict.get("omissions", []):
        if isinstance(issue, dict):
            omissions.append(issue.get("instruction_to_refiner") or issue.get("missing_fact") or json.dumps(issue))
    return {
        "faithful": verdict.get("overall_decision") == "accept",
        "unsupported_claims": unsupported,
        "omissions": omissions,
        "production_verdict": verdict,
    }


async def _default_agent_executor(agent: Any, state: dict[str, Any], message: str,
                                  on_text: Callable[[str, str], None]) -> str:
    return await run_agent_stream(agent, state=state, message=message, on_text=on_text)


async def _default_verifier_runner(source_text: str, facts: str, draft: str, model: str,
                                   on_text: Callable[[str, str], None], definitions: str) -> str:
    return await run_production_verifier(
        source_text, facts, draft, model, on_text=on_text, reference_definitions=definitions,
    )


async def run_production_pipeline(
    source_text: str,
    model: str = VERIFIER_DEFAULT_MODEL,
    *,
    on_event: TraceCallback | None = None,
    agent_executor: AgentExecutor = _default_agent_executor,
    verifier_runner: VerifierRunner = _default_verifier_runner,
    lookup: MedicalLookup | None = None,
) -> PipelineRunResult:
    """Run every production stage while preserving an inspectable in-memory trace.

    ``lookup`` supplies UMLS reference definitions; by default it is
    configured from the environment. Lookup problems never stop the pipeline.
    """
    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("source_text must be a nonempty medical report.")
    lookup = lookup if lookup is not None else MedicalLookup.from_env()
    trace: dict[str, Any] = {
        "schema": "faithfulmed_production_trace_v2",
        "started_at_utc": _now(),
        "completed_at_utc": None,
        "model": model,
        "status": "running",
        "original_report": source_text,
        "stages": [],
        "results": {"extractor": None, "reference_definitions": None, "simplifier": None, "verifier_refiner_iterations": [],
                    "simplifier_review_flags": None, "readability": None, "final_explanation": None},
        "error": None,
    }

    async def stage(stage_id: str, label: str, stage_input: dict[str, Any], agent: Any,
                    state: dict[str, Any], message: str, iteration: int | None = None) -> str:
        record = {"stage_id": stage_id, "agent": label, "iteration": iteration, "status": "running",
                  "started_at_utc": _now(), "completed_at_utc": None, "input": stage_input,
                  "output": "", "error": None}
        trace["stages"].append(record)
        _emit(on_event, {"type": "stage_started", "record": record})

        def on_text(delta: str, accumulated: str) -> None:
            record["output"] = accumulated
            _emit(on_event, {"type": "stage_chunk", "stage_id": stage_id, "agent": label,
                             "iteration": iteration, "delta": delta, "output": accumulated})

        try:
            output = await agent_executor(agent, state, message, on_text)
        except Exception as exc:
            record.update(status="failed", completed_at_utc=_now(), error=str(exc))
            _emit(on_event, {"type": "stage_failed", "record": record})
            raise
        record.update(status="completed", completed_at_utc=_now(), output=output)
        _emit(on_event, {"type": "stage_completed", "record": record})
        return output

    async def verify(draft: str, check_number: int) -> tuple[str, dict[str, Any]]:
        stage_id = f"verifier-{check_number}"
        stage_input = {"original_report": source_text, "extracted_facts": json.loads(atoms),
                       "reference_definitions": verifier_definitions,
                       "patient_friendly_explanation_draft": draft}
        record = {"stage_id": stage_id, "agent": "Verifier", "iteration": check_number,
                  "status": "running", "started_at_utc": _now(), "completed_at_utc": None,
                  "input": stage_input, "output": "", "error": None}
        trace["stages"].append(record)
        _emit(on_event, {"type": "stage_started", "record": record})

        def on_text(delta: str, accumulated: str) -> None:
            record["output"] = accumulated
            _emit(on_event, {"type": "stage_chunk", "stage_id": stage_id, "agent": "Verifier",
                             "iteration": check_number, "delta": delta, "output": accumulated})

        try:
            raw = await verifier_runner(source_text, atoms, draft, model, on_text,
                                        json.dumps(verifier_definitions, ensure_ascii=False))
            parsed = parse_production_verdict(raw)
        except Exception as exc:
            record.update(status="failed", completed_at_utc=_now(), error=str(exc))
            _emit(on_event, {"type": "stage_failed", "record": record})
            raise
        record.update(status="completed", completed_at_utc=_now(), output=raw,
                      parsed_output=parsed)
        _emit(on_event, {"type": "stage_completed", "record": record})
        return raw, parsed

    async def term_lookup() -> dict[str, Any]:
        """Pick terms and look them up. Records failures; never raises."""
        record = {"stage_id": "term_lookup", "agent": "TermLookup", "iteration": None,
                  "status": "running", "started_at_utc": _now(), "completed_at_utc": None,
                  "input": {"extracted_facts": json.loads(atoms), "sources": lookup.sources()},
                  "output": "", "error": None}
        trace["stages"].append(record)
        _emit(on_event, {"type": "stage_started", "record": record})

        def on_text(delta: str, accumulated: str) -> None:
            record["output"] = accumulated
            _emit(on_event, {"type": "stage_chunk", "stage_id": "term_lookup", "agent": "TermLookup",
                             "iteration": None, "delta": delta, "output": accumulated})

        async def find_terms(facts_json: str) -> str:
            return await agent_executor(create_term_finder_agent(model), {"atoms": facts_json},
                                        "List the terms to look up.", on_text)

        try:
            definitions = await build_reference_definitions(atoms, lookup, find_terms)
        except Exception as exc:  # build_reference_definitions should not raise; stay safe
            definitions = empty_reference_definitions("failed", lookup, [f"{type(exc).__name__}: {exc}"])
        status = {"completed": "completed", "not_configured": "skipped"}.get(definitions["status"], "failed")
        record.update(status=status, completed_at_utc=_now(),
                      output=json.dumps(definitions, ensure_ascii=False), parsed_output=definitions,
                      error="; ".join(definitions["errors"]) or None)
        _emit(on_event, {"type": "stage_completed", "record": record})
        return definitions

    try:
        atoms_raw = await stage(
            "extractor", "Extractor", {"original_report": source_text},
            create_extractor_agent(model), {"source_text": source_text},
            "Extract the facts from the supplied medical source.",
        )
        atoms = _strip_json_fence(atoms_raw)
        ExtractorOutput.model_validate_json(atoms)
        trace["results"]["extractor"] = atoms

        # Reference definitions stay separate from the patient-specific facts.
        definitions = await term_lookup()
        trace["results"]["reference_definitions"] = definitions
        glossary = render_glossary(definitions)
        verifier_definitions = reference_definitions_for_prompt(definitions)

        draft = await stage(
            "simplifier", "Simplifier", {"original_report": source_text, "extracted_facts": json.loads(atoms),
                                          "glossary": glossary},
            create_simplifier_agent(model),
            {"source_text": source_text, "atoms": atoms, "glossary": glossary},
            "Create the patient-friendly medical text.",
        )
        trace["results"]["simplifier"] = draft  # raw output, both sections
        draft, trace["results"]["simplifier_review_flags"] = split_simplifier_output(draft)

        accepted = False
        revisions = 0
        check_number = 1
        while True:
            raw_verdict, verdict = await verify(draft, check_number)
            exchange: dict[str, Any] = {
                "verification_number": check_number, "revision_cycle": revisions,
                "draft_input": draft, "verifier_output": raw_verdict,
                "parsed_verdict": verdict, "refiner_input": None, "refiner_output": None,
            }
            trace["results"]["verifier_refiner_iterations"].append(exchange)
            if verdict["overall_decision"] == "accept":
                accepted = True
                break
            if revisions >= MAX_REVISION_CYCLES:
                break
            revisions += 1
            feedback = refiner_feedback(verdict)
            exchange["refiner_input"] = {"original_report": source_text, "draft": draft,
                                           "verifier_feedback": feedback}
            draft = await stage(
                f"refiner-{revisions}", "Refiner", exchange["refiner_input"],
                create_refiner_agent(model),
                {"source_text": source_text, "draft": draft,
                 "verdict": json.dumps(feedback, ensure_ascii=False)},
                "Refine the draft using the supplied feedback.", iteration=revisions,
            )
            exchange["refiner_output"] = draft
            check_number += 1

        if not accepted:
            trace.update(status="not_accepted", completed_at_utc=_now())
            trace["latest_draft"] = draft
            _emit(on_event, {"type": "pipeline_not_accepted", "trace": trace})
            return PipelineRunResult(trace, None, False)

        final = await stage(
            "readability", "Readability", {"accepted_draft": draft},
            create_readability_agent(model), {"draft": draft},
            "Return the improved the draft.",
        )
        trace["results"]["readability"] = final
        trace["results"]["final_explanation"] = final
        trace.update(status="completed", completed_at_utc=_now())
        _emit(on_event, {"type": "pipeline_completed", "trace": trace})
        return PipelineRunResult(trace, final, True)
    except Exception as exc:
        trace.update(status="failed", completed_at_utc=_now(), error=str(exc))
        _emit(on_event, {"type": "pipeline_failed", "trace": trace})
        return PipelineRunResult(trace, None, False)
