"""Smoke test: UMLS reference definitions through the production pipeline.

Runs ``run_production_pipeline`` on one synthetic note, captures exactly what each
agent receives, and checks four steps:

  1  The Extractor produces a hypertension fact with source_evidence from the note.
  2  The lookup returns UMLS C0020538 linked to that fact.
  3a The Simplifier gets the reference in its glossary, separate from the facts,
     with the fact link.
  3b The Verifier gets the reference as its own input key, separate from the facts.
  4  Definition quality: printed for manual review, with simple relevance hints.

Modes:
  mock       Offline. Gemini stages and UMLS are mocked. Run by the unit tests.
  umls-live  Real UMLS (needs UMLS_API_KEY); Gemini stages mocked.
  live       Real UMLS and Gemini (needs UMLS_API_KEY and GOOGLE_API_KEY). Uses quota.
  auto       live if both keys are set, else umls-live if UMLS_API_KEY is set, else mock.

    python3 -m evals.lookup_smoke --mode umls-live
    python3 -m evals.lookup_smoke --mode live --model <gemini model id>

``--model`` sets the model for every Gemini stage (Extractor, TermFinder,
Simplifier, Verifier, Refiner, Readability). Without it the pipeline default is used.

The UMLS key is read only from the environment and is masked in all output.
Nothing is written to disk.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import urllib.parse
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import patch

from agents import production_pipeline as pp
from agents.medical_lookup import MedicalLookup, UMLSClient
from agents.verifier import production_verifier_prompt

NOTE = "The patient has a history of hypertension. Continue the current treatment plan."
CUI = "C0020538"
MODES = ("auto", "live", "umls-live", "mock")

# ── Mocks (used only where the mode requires them) ─────────────────────
MOCK_FACTS = json.dumps([
    {"fact_id": "F-001", "fact_text": "The patient has a history of hypertension.", "fact_type": "diagnosis",
     "source_evidence": ["The patient has a history of hypertension."]},
    {"fact_id": "F-002", "fact_text": "Continue the current treatment plan.", "fact_type": "instruction",
     "source_evidence": ["Continue the current treatment plan."]},
])
MOCK_TERMS = json.dumps([{"term": "hypertension", "category": "clinical_concept", "fact_ids": ["F-001"]}])
MOCK_DEFINITION = "[MOCK definition: not real UMLS content]"
ACCEPT = json.dumps({"overall_decision": "accept", "explanation_accuracy": {"prediction": "yes", "reason": "mock"},
                     "sentences": [], "unsupported_claims": [], "omissions": []})
MOCK_TEXT = {"Extractor": MOCK_FACTS, "TermFinder": MOCK_TERMS,
             "Simplifier": "PATIENT-FRIENDLY VERSION:\nMock draft.\n\nREVIEW FLAGS:\n- None.",
             "Readability": "Mock final."}
AGENT_FACTORIES = ("create_extractor_agent", "create_term_finder_agent", "create_simplifier_agent",
                   "create_refiner_agent", "create_readability_agent")


def _fake_uts(url: str) -> Dict[str, Any]:
    path = urllib.parse.urlparse(url).path
    if "/search/" in path:
        return {"result": {"results": [{"ui": CUI, "name": "Hypertensive disease"}]}}
    return {"result": [{"rootSource": "MSH", "value": MOCK_DEFINITION}]}


def resolve_mode(mode: str, env: Dict[str, str]) -> str:
    has_umls = bool(env.get("UMLS_API_KEY"))
    has_gemini = bool(env.get("GOOGLE_API_KEY")) or env.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
    if mode == "auto":
        return "live" if has_umls and has_gemini else "umls-live" if has_umls else "mock"
    if mode in ("live", "umls-live") and not has_umls:
        raise RuntimeError("BLOCKED: UMLS_API_KEY is not set in this environment.")
    if mode == "live" and not has_gemini:
        raise RuntimeError("BLOCKED: GOOGLE_API_KEY (or Vertex config) is not set; use --mode umls-live.")
    return mode


async def run_smoke(mode: str = "mock", env: Dict[str, str] | None = None,
                    model: str | None = None) -> Dict[str, Any]:
    """Run the pipeline once and return the evidence and a verdict for each step."""
    env = dict(os.environ) if env is None else env
    mode = resolve_mode(mode, env)
    llm_live, umls_live = mode == "live", mode in ("live", "umls-live")
    lookup = MedicalLookup.from_env(env) if umls_live else MedicalLookup(umls=UMLSClient("mock-key", fetch=_fake_uts))
    captured: Dict[str, list] = {}

    async def executor(agent, state, message, on_text):
        captured.setdefault(agent.name, []).append(dict(state))
        if llm_live:
            return await pp._default_agent_executor(agent, state, message, on_text)
        text = MOCK_TEXT[agent.name]
        on_text(text, text)
        return text

    async def verifier_runner(source, facts, draft, model, on_text, definitions):
        captured.setdefault("Verifier", []).append(
            {"prompt": production_verifier_prompt(source, facts, draft, definitions)})
        if llm_live:
            return await pp._default_verifier_runner(source, facts, draft, model, on_text, definitions)
        return ACCEPT

    with contextlib.ExitStack() as stack:
        if not llm_live:  # stand-in agents: no ADK objects, no model calls
            names = {"create_extractor_agent": "Extractor", "create_term_finder_agent": "TermFinder",
                     "create_simplifier_agent": "Simplifier", "create_refiner_agent": "Refiner",
                     "create_readability_agent": "Readability"}
            for factory in AGENT_FACTORIES:
                stack.enter_context(patch.object(pp, factory, return_value=SimpleNamespace(name=names[factory])))
        model_kw = {"model": model} if model else {}
        result = await pp.run_production_pipeline(NOTE, agent_executor=executor,
                                                  verifier_runner=verifier_runner, lookup=lookup, **model_kw)

    trace = result.trace
    steps: Dict[str, Dict[str, Any]] = {}

    # Step 1: Extractor fact grounded in the note
    facts = json.loads(trace["results"]["extractor"] or "[]")
    htn = [f for f in facts if "hypertension" in f["fact_text"].lower()]
    htn_ids = [f["fact_id"] for f in htn]
    steps["1"] = {"pass": bool(htn) and all(
        f["source_evidence"] and all(ev in NOTE for ev in f["source_evidence"])
        and any("hypertension" in ev.lower() for ev in f["source_evidence"]) for f in htn),
        "evidence": {"facts": facts}}

    # Step 2: UMLS reference linked to that fact
    defs = trace["results"]["reference_definitions"] or {}
    entry = next((e for e in defs.get("entries", []) if e.get("source") == "UMLS" and e.get("source_id") == CUI), None)
    steps["2"] = {"pass": bool(entry) and bool(set(entry["fact_ids"]) & set(htn_ids)),
                  "evidence": {k: defs.get(k) for k in ("status", "entries", "unresolved", "dropped_terms", "errors")}}

    # Step 3a: Simplifier handoff
    simp = (captured.get("Simplifier") or [{}])[0]
    gloss = simp.get("glossary", "")
    simp_atoms = simp.get("atoms", "[]")
    line = next((ln for ln in gloss.splitlines() if CUI in ln), "")
    checks_a = {
        "glossary has CUI": bool(line),
        "glossary has definition text": bool(entry) and entry["definition"] in gloss,
        "glossary line names the fact": bool(htn_ids) and any(f"facts {i}" in line or f", {i}" in line for i in htn_ids),
        "facts handed over unchanged, no reference inside": json.loads(simp_atoms) == facts and CUI not in simp_atoms,
        "fact keeps source_evidence": any(f["fact_id"] in htn_ids and f["source_evidence"] for f in json.loads(simp_atoms)),
    }
    steps["3a"] = {"pass": all(checks_a.values()), "checks": checks_a, "evidence": {"glossary": gloss}}

    # Step 3b: Verifier handoff
    vprompt = (captured.get("Verifier") or [{}])[0].get("prompt", "")
    payload = json.loads(vprompt.rsplit("INPUT JSON:\n", 1)[1]) if "INPUT JSON:\n" in vprompt else {}
    vrefs = payload.get("reference_definitions") or []
    vfacts = payload.get("extracted_facts") or []
    checks_b = {
        "reference_definitions key present": bool(vrefs),
        "reference has CUI and fact link": any(
            r.get("source_id") == CUI and set(r.get("fact_ids") or []) & set(htn_ids) for r in vrefs),
        "facts handed over unchanged, no reference inside": vfacts == facts and CUI not in json.dumps(vfacts),
        "fact keeps source_evidence": any(f["fact_id"] in htn_ids and f["source_evidence"] for f in vfacts),
        "prompt has REFERENCE DEFINITIONS rules": "REFERENCE DEFINITIONS" in vprompt,
    }
    steps["3b"] = {"pass": all(checks_b.values()), "checks": checks_b,
                   "evidence": {"payload_keys": list(payload), "reference_definitions": vrefs}}

    # Step 4: definition quality (needs live UMLS to mean anything)
    quality: Dict[str, Any] = {}
    if entry:
        text = entry["definition"].lower()
        quality = {k: entry.get(k) for k in ("matched_name", "definition_source", "truncated", "definition")}
        quality["hints"] = {"mentions hypertension": "hypertension" in text,
                            "mentions high/elevated blood pressure": any(
                                p in text for p in ("high blood pressure", "high systemic arterial blood pressure",
                                                    "elevated blood pressure", "high arterial blood pressure")),
                            "length": len(entry["definition"])}
    steps["4"] = {"pass": "manual review" if umls_live else "blocked (mock text)", "evidence": quality}

    return {"mode": mode, "model": trace["model"], "llm_stages": "live" if llm_live else "mocked", "umls": "live" if umls_live else "mocked",
            "pipeline_status": trace["status"], "pipeline_error": trace["error"],
            "accepted": result.accepted, "final_explanation": result.final_explanation,
            "stages": [(s["agent"], s["status"]) for s in trace["stages"]], "steps": steps}


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="FaithfulMed UMLS lookup smoke test.")
    parser.add_argument("--mode", choices=MODES, default="auto")
    parser.add_argument("--model", default=None, help="Gemini model ID for all LLM stages (live mode).")
    args = parser.parse_args(argv)
    secret = os.environ.get("UMLS_API_KEY", "")
    try:
        report = asyncio.run(run_smoke(args.mode, model=args.model))
    except RuntimeError as exc:
        print(_redact(str(exc), secret))
        return 2
    print(_redact(json.dumps(report, indent=2, ensure_ascii=False), secret))
    summary = {k: ("PASS" if v["pass"] is True else "FAIL" if v["pass"] is False else v["pass"])
               for k, v in report["steps"].items()}
    print(f"\nMODE {report['mode']} (model {report['model']}, Gemini stages {report['llm_stages']}, UMLS {report['umls']})")
    print("SUMMARY", json.dumps(summary))
    return 0 if all(v["pass"] is not False for v in report["steps"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
