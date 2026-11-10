"""Offline tests for UMLS/DrugBank reference definitions. No network, no Gemini."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.extractor import ExtractorOutput
from agents.medical_lookup import (
    NO_GLOSSARY, REFERENCE_DEFINITIONS_SCHEMA, TERM_FINDER_INSTRUCTION, DrugBankIndex,
    LookupServiceError, MedicalLookup, UMLSClient, build_drugbank_index, build_reference_definitions,
    lookup_config_from_env, parse_term_candidates, reference_definitions_for_prompt, render_glossary,
)
from agents.production_pipeline import run_production_pipeline
from agents.verifier import production_verifier_prompt

KEY = "secret-uts-key"

FACTS = [
    {"fact_id": "F-001", "fact_text": "Continue metoprolol 25 mg b.i.d. for hypertension.",
     "fact_type": "medication", "source_evidence": ["Continue metoprolol 25 mg b.i.d. for hypertension."]},
    {"fact_id": "F-002", "fact_text": "History of hypertension.", "fact_type": "diagnosis",
     "source_evidence": ["History of hypertension."]},
]
SOURCE = "Continue metoprolol 25 mg b.i.d. for hypertension. History of hypertension."

DRUGBANK_XML = """<?xml version="1.0" encoding="UTF-8"?>
<drugbank xmlns="http://www.drugbank.ca" version="5.1.99" exported-on="2026-01-01">
  <drug type="small molecule">
    <drugbank-id primary="true">DB00264</drugbank-id>
    <drugbank-id>APRD00059</drugbank-id>
    <name>Metoprolol</name>
    <description>Metoprolol is a beta-blocker used to treat high blood pressure.[A1234, L5678] It is taken by mouth.[FDA Label]</description>
    <synonyms><synonym language="english">Metoprololum</synonym></synonyms>
    <products><product><name>Lopressor</name></product></products>
    <salts><salt><drugbank-id>DBSALT000001</drugbank-id><name>Metoprolol tartrate</name></salt></salts>
    <drug-interactions>
      <drug-interaction><drugbank-id>DB00999</drugbank-id><name>Nestedonly</name></drug-interaction>
    </drug-interactions>
  </drug>
  <drug type="small molecule">
    <drugbank-id primary="true">DB00001</drugbank-id>
    <name>Alphadrug</name>
    <description></description>
    <products><product><name>Sharedbrand</name></product></products>
  </drug>
  <drug type="small molecule">
    <drugbank-id primary="true">DB00002</drugbank-id>
    <name>Betadrug</name>
    <description>Betadrug description.</description>
    <products><product><name>Sharedbrand</name></product></products>
  </drug>
</drugbank>
"""


def fake_umls(concepts: dict, definitions: dict, calls: list | None = None):
    """concepts: term(lower) -> list of (cui, name); definitions: cui -> list of (rootSource, value)."""
    def fetch(url: str):
        parsed = urllib.parse.urlparse(url)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        assert params.get("apiKey") == KEY
        if calls is not None:
            calls.append(parsed.path)
        if "/search/" in parsed.path:
            hits = concepts.get(params["string"].lower(), [])
            if not hits:
                return {"result": {"results": [{"ui": "NONE", "name": "NO RESULTS"}]}}
            return {"result": {"results": [{"ui": c, "name": n} for c, n in hits]}}
        cui = parsed.path.split("/CUI/")[1].split("/")[0]
        if cui not in definitions:
            return None  # UTS answers 404 when a concept has no definitions
        return {"result": [{"rootSource": s, "value": v} for s, v in definitions[cui]]}
    return fetch


def build_index(tmp: str) -> DrugBankIndex:
    xml = Path(tmp) / "db.xml"
    xml.write_text(DRUGBANK_XML, encoding="utf-8")
    out = Path(tmp) / "index.json"
    build_drugbank_index(xml, out)
    return DrugBankIndex.load(out)


class UMLSClientTests(unittest.TestCase):
    def client(self, **kw):
        return UMLSClient(KEY, fetch=fake_umls(**kw))

    def test_concise_definition_beats_long_medlineplus_summary(self) -> None:
        # MedlinePlus-shaped: a long topic summary whose first 600 characters only
        # describe blood pressure in general (the failure seen for hypertension).
        background = ("<h3>What is blood pressure?</h3><p>Blood pressure is the force of your blood "
                      "pushing against the walls of your arteries. " + "Background sentence. " * 40 + "</p>")
        client = self.client(
            concepts={"hypertension": [("C0020538", "Hypertensive disease")]},
            definitions={"C0020538": [("SNOMEDCT_US", "Ignored."), ("MEDLINEPLUS", background),
                                      ("MSH", "Persistently high systemic arterial blood pressure.")]},
        )
        result = client.lookup("hypertension")
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["source_id"], "C0020538")
        self.assertEqual(result["definition_source"], "MSH")
        self.assertEqual(result["definition"], "Persistently high systemic arterial blood pressure.")
        self.assertFalse(result["truncated"])
        self.assertEqual(result["matched_name"], "Hypertensive disease")
        self.assertIn("C0020538", result["reference_url"])

    def test_medlineplus_is_last_resort_and_markup_is_cleaned(self) -> None:
        client = self.client(
            concepts={"hypertension": [("C0020538", "Hypertensive disease")]},
            definitions={"C0020538": [("SNOMEDCT_US", "Ignored."),
                                      ("MEDLINEPLUS", "<p>High blood pressure &amp; strain.</p>")]},
        )
        result = client.lookup("hypertension")
        self.assertEqual(result["definition_source"], "MEDLINEPLUS")
        self.assertEqual(result["definition"], "High blood pressure & strain.")

    def test_not_found(self) -> None:
        result = self.client(concepts={}, definitions={}).lookup("zzzz")
        self.assertEqual((result["status"], result["reason"]), ("miss", "not_found"))

    def test_concept_without_allowed_definition_is_no_definition(self) -> None:
        client = self.client(concepts={"thing": [("C1", "Thing")]},
                             definitions={"C1": [("SNOMEDCT_US", "Restricted source.")]})
        result = client.lookup("thing")
        self.assertEqual(result["reason"], "no_definition")
        self.assertNotIn("definition", result)
        self.assertEqual(self.client(concepts={"x": [("C2", "X")]}, definitions={}).lookup("x")["reason"],
                         "no_definition")

    def test_ambiguous_abbreviation_is_not_resolved(self) -> None:
        client = self.client(concepts={"ms": [("C0026769", "Multiple sclerosis"), ("C0026269", "Mitral stenosis")]},
                             definitions={"C0026769": [("MSH", "Def.")]})
        self.assertEqual(client.lookup("MS")["reason"], "ambiguous")

    def test_results_are_cached(self) -> None:
        calls: list = []
        client = UMLSClient(KEY, fetch=fake_umls({"gout": [("C1", "Gout")]}, {"C1": [("MSH", "Def.")]}, calls))
        client.lookup("gout")
        client.lookup("Gout")
        self.assertEqual(len(calls), 2)  # one search + one definitions call

    def test_service_error_never_leaks_key(self) -> None:
        def broken(url):
            raise LookupServiceError(f"failed for {url}")
        lookup = MedicalLookup(umls=UMLSClient(KEY, fetch=broken))
        result = lookup.lookup_candidate({"term": "gout", "category": "clinical_concept", "fact_ids": ["F-1"]})
        self.assertFalse(result["found"])
        attempt = result["unresolved"]["attempts"][0]
        self.assertEqual(attempt["reason"], "api_error")
        self.assertNotIn(KEY, json.dumps(result))


class DrugBankIndexTests(unittest.TestCase):
    def test_index_lookup_by_name_brand_synonym_and_salt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = build_index(tmp)
        for name in ("metoprolol", "Lopressor", "METOPROLOLUM", "metoprolol  tartrate"):
            with self.subTest(name=name):
                result = index.lookup(name)
                self.assertEqual(result["status"], "found")
                self.assertEqual(result["source_id"], "DB00264")
        result = index.lookup("metoprolol")
        self.assertEqual(result["definition"],
                         "Metoprolol is a beta-blocker used to treat high blood pressure. It is taken by mouth.")
        self.assertEqual(result["source_version"], "5.1.99")
        self.assertEqual(result["reference_url"], "https://go.drugbank.com/drugs/DB00264")

    def test_nested_names_are_not_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = build_index(tmp)
        self.assertEqual(index.lookup("Nestedonly")["reason"], "not_found")
        self.assertEqual(index.lookup("APRD00059")["reason"], "not_found")

    def test_ambiguous_and_missing_description(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = build_index(tmp)
        self.assertEqual(index.lookup("Sharedbrand")["reason"], "ambiguous")
        self.assertEqual(index.lookup("Alphadrug")["reason"], "no_definition")

    def test_long_definition_is_shortened_at_sentence(self) -> None:
        index = DrugBankIndex({"schema": "faithfulmed_drugbank_index_v1", "drugs": {
            "DB1": {"name": "Long", "description": ("First sentence is here. " * 60)}},
            "primary_names": {"long": ["DB1"]}, "aliases": {}})
        result = index.lookup("long")
        self.assertTrue(result["truncated"])
        self.assertLessEqual(len(result["definition"]), 600)
        self.assertTrue(result["definition"].endswith("."))


class TermCandidateTests(unittest.TestCase):
    def test_terms_must_appear_in_cited_fact_text(self) -> None:
        raw = json.dumps([
            {"term": "metoprolol", "category": "medication", "fact_ids": ["F-001"]},
            {"term": "Hypertension", "category": "clinical_concept", "fact_ids": ["F-001", "F-002", "F-999"]},
            {"term": "hypertension", "category": "clinical_concept", "fact_ids": ["F-002"]},
            {"term": "high blood pressure", "category": "clinical_concept", "fact_ids": ["F-002"]},
            {"term": "metop", "category": "medication", "fact_ids": ["F-001"]},
            {"term": "aspirin", "category": "drug", "fact_ids": ["F-001"]},
            "not an object",
        ])
        candidates, dropped = parse_term_candidates(f"```json\n{raw}\n```", FACTS)
        self.assertEqual(candidates, [
            {"term": "metoprolol", "category": "medication", "fact_ids": ["F-001"]},
            {"term": "Hypertension", "category": "clinical_concept", "fact_ids": ["F-001", "F-002"]},
        ])
        reasons = {d.get("term"): d["reason"] for d in dropped}
        self.assertEqual(reasons["high blood pressure"], "term_not_in_cited_fact_text")
        self.assertEqual(reasons["metop"], "term_not_in_cited_fact_text")  # no partial-word matches
        self.assertEqual(reasons["aspirin"], "invalid_category")

    def test_invalid_output_raises(self) -> None:
        for raw in ("not json", '{"term": "x"}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_term_candidates(raw, FACTS)

    def test_instruction_has_only_atoms_placeholder(self) -> None:
        self.assertEqual(TERM_FINDER_INSTRUCTION.count("{"), 1)
        self.assertIn("{atoms}", TERM_FINDER_INSTRUCTION)


def configured_lookup(tmp: str) -> MedicalLookup:
    umls = UMLSClient(KEY, fetch=fake_umls(
        {"hypertension": [("C0020538", "Hypertensive disease")]},
        {"C0020538": [("MSH", "Persistently high systemic arterial blood pressure.")]},
    ))
    return MedicalLookup(umls=umls, drugbank=build_index(tmp))


TERMS = json.dumps([
    {"term": "metoprolol", "category": "medication", "fact_ids": ["F-001"]},
    {"term": "hypertension", "category": "clinical_concept", "fact_ids": ["F-001", "F-002"]},
    {"term": "b.i.d", "category": "clinical_concept", "fact_ids": ["F-001"]},
])


class BuildReferenceDefinitionsTests(unittest.TestCase):
    def run_build(self, lookup, find_terms):
        return asyncio.run(build_reference_definitions(json.dumps(FACTS), lookup, find_terms))

    def test_successful_lookup_routes_to_the_right_source(self) -> None:
        async def find_terms(atoms):
            return TERMS
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_build(configured_lookup(tmp), find_terms)
        self.assertEqual(result["schema"], REFERENCE_DEFINITIONS_SCHEMA)
        self.assertEqual(result["status"], "completed")
        by_term = {e["term"]: e for e in result["entries"]}
        self.assertEqual((by_term["metoprolol"]["source"], by_term["metoprolol"]["source_id"]), ("DrugBank", "DB00264"))
        self.assertEqual((by_term["hypertension"]["source"], by_term["hypertension"]["source_id"]), ("UMLS", "C0020538"))
        for entry in result["entries"]:
            for field in ("term", "definition", "source", "source_id", "reference_url", "fact_ids"):
                self.assertTrue(entry[field], field)
        self.assertEqual([u["term"] for u in result["unresolved"]], ["b.i.d"])
        self.assertEqual(result["unresolved"][0]["attempts"][0]["reason"], "not_found")

    def test_medication_falls_back_to_umls_when_drugbank_missing(self) -> None:
        umls = UMLSClient(KEY, fetch=fake_umls({"metoprolol": [("C0025859", "Metoprolol")]},
                                               {"C0025859": [("NCI", "A beta blocker.")]}))
        result = MedicalLookup(umls=umls).lookup_candidate(
            {"term": "metoprolol", "category": "medication", "fact_ids": ["F-001"]})
        self.assertTrue(result["found"])
        self.assertEqual(result["entry"]["source"], "UMLS")

    def test_not_configured_skips_model_call(self) -> None:
        async def find_terms(atoms):
            raise AssertionError("TermFinder must not run without a lookup source")
        result = self.run_build(MedicalLookup.disabled(), find_terms)
        self.assertEqual(result["status"], "not_configured")
        self.assertEqual(result["entries"], [])
        self.assertEqual(render_glossary(result), NO_GLOSSARY)

    def test_term_finder_failure_is_recorded_without_definitions(self) -> None:
        async def find_terms(atoms):
            raise RuntimeError("model unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_build(configured_lookup(tmp), find_terms)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["entries"], [])
        self.assertIn("model unavailable", result["errors"][0])

    def test_term_limit(self) -> None:
        async def find_terms(atoms):
            return TERMS
        with tempfile.TemporaryDirectory() as tmp:
            lookup = configured_lookup(tmp)
            lookup.max_terms = 1
            result = self.run_build(lookup, find_terms)
        self.assertEqual(len(result["entries"]) + len(result["unresolved"]), 1)
        self.assertEqual(sum(d["reason"] == "term_limit" for d in result["dropped_terms"]), 2)

    def test_env_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {"UMLS_API_KEY": "", "DRUGBANK_INDEX_PATH": str(Path(tmp) / "missing.json")}
            self.assertFalse(MedicalLookup.from_env(env).configured)
            self.assertEqual(lookup_config_from_env(env), {"umls": False, "drugbank_index": None})
            bad = Path(tmp) / "bad.json"
            bad.write_text("{}", encoding="utf-8")
            lookup = MedicalLookup.from_env({"UMLS_API_KEY": KEY, "DRUGBANK_INDEX_PATH": str(bad)})
            self.assertIsNotNone(lookup.umls)
            self.assertIsNone(lookup.drugbank)
            self.assertTrue(lookup.config_errors)


class DownstreamFormatTests(unittest.TestCase):
    def definitions(self):
        async def find_terms(atoms):
            return TERMS
        with tempfile.TemporaryDirectory() as tmp:
            return asyncio.run(build_reference_definitions(json.dumps(FACTS), configured_lookup(tmp), find_terms))

    def test_glossary_carries_definition_and_provenance(self) -> None:
        glossary = render_glossary(self.definitions())
        self.assertIn("not facts about this patient", glossary)
        self.assertIn("metoprolol [DrugBank DB00264", glossary)
        self.assertIn("hypertension [UMLS C0020538", glossary)
        self.assertIn("Persistently high systemic arterial blood pressure.", glossary)
        lines = {line.split(" [", 1)[0]: line for line in glossary.splitlines() if line.startswith("- ")}
        self.assertIn("for facts F-001]", lines["- metoprolol"])
        self.assertIn("for facts F-001, F-002]", lines["- hypertension"])

    def test_verifier_gets_definitions_as_a_separate_key(self) -> None:
        compact = reference_definitions_for_prompt(self.definitions())
        rendered = production_verifier_prompt(SOURCE, json.dumps(FACTS), "Draft", json.dumps(compact))
        payload = json.loads(rendered.rsplit("INPUT JSON:\n", 1)[1])
        self.assertEqual(payload["extracted_facts"], FACTS)
        self.assertEqual({d["source_id"] for d in payload["reference_definitions"]}, {"DB00264", "C0020538"})
        empty = production_verifier_prompt(SOURCE, json.dumps(FACTS), "Draft", "[]")
        self.assertNotIn("reference_definitions", json.loads(empty.rsplit("INPUT JSON:\n", 1)[1]))


ACCEPT = json.dumps({"overall_decision": "accept", "explanation_accuracy": {"prediction": "yes", "reason": "ok"},
                     "sentences": [], "unsupported_claims": [], "omissions": []})


def stub(name):
    return SimpleNamespace(name=name)


@patch("agents.production_pipeline.create_readability_agent", return_value=stub("Readability"))
@patch("agents.production_pipeline.create_simplifier_agent", return_value=stub("Simplifier"))
@patch("agents.production_pipeline.create_term_finder_agent", return_value=stub("TermFinder"))
@patch("agents.production_pipeline.create_extractor_agent", return_value=stub("Extractor"))
class PipelineIntegrationTests(unittest.TestCase):
    def run_pipeline(self, lookup, term_output):
        seen: dict = {}
        extracted = json.dumps(FACTS)

        async def executor(agent, state, message, on_text):
            seen[agent.name] = state
            if agent.name == "TermFinder" and isinstance(term_output, Exception):
                raise term_output
            text = {"Extractor": extracted, "TermFinder": term_output,
                    "Simplifier": "Draft.", "Readability": "Final."}[agent.name]
            on_text(text, text)
            return text

        async def verifier_runner(source, facts, draft, model, on_text, definitions):
            seen["Verifier"] = {"facts": facts, "definitions": json.loads(definitions)}
            return ACCEPT

        result = asyncio.run(run_production_pipeline(
            SOURCE, agent_executor=executor, verifier_runner=verifier_runner, lookup=lookup))
        return result, seen, extracted

    def test_definitions_reach_simplifier_and_verifier_and_facts_are_unchanged(self, *_):
        with tempfile.TemporaryDirectory() as tmp:
            result, seen, extracted = self.run_pipeline(configured_lookup(tmp), TERMS)
        self.assertTrue(result.accepted)
        self.assertEqual(result.trace["results"]["extractor"], extracted)
        ExtractorOutput.model_validate_json(result.trace["results"]["extractor"])
        self.assertEqual(json.loads(seen["Verifier"]["facts"]), FACTS)  # no definitions mixed into facts
        self.assertEqual(seen["Simplifier"]["atoms"], extracted)
        self.assertIn("DrugBank DB00264", seen["Simplifier"]["glossary"])
        self.assertEqual({d["source_id"] for d in seen["Verifier"]["definitions"]}, {"DB00264", "C0020538"})
        self.assertEqual(seen["TermFinder"], {"atoms": extracted})
        stage = next(s for s in result.trace["stages"] if s["agent"] == "TermLookup")
        self.assertEqual(stage["status"], "completed")
        self.assertEqual(result.trace["results"]["reference_definitions"]["status"], "completed")

    def test_lookup_failure_does_not_interrupt_pipeline(self, *_):
        with tempfile.TemporaryDirectory() as tmp:
            result, seen, _ = self.run_pipeline(configured_lookup(tmp), RuntimeError("quota"))
        self.assertTrue(result.accepted)
        self.assertEqual(result.final_explanation, "Final.")
        self.assertEqual(seen["Simplifier"]["glossary"], NO_GLOSSARY)
        self.assertEqual(seen["Verifier"]["definitions"], [])
        stage = next(s for s in result.trace["stages"] if s["agent"] == "TermLookup")
        self.assertEqual(stage["status"], "failed")
        self.assertIn("quota", stage["error"])

    def test_api_errors_become_unresolved_terms(self, *_):
        def down(url):
            raise LookupServiceError("HTTP 503")
        lookup = MedicalLookup(umls=UMLSClient(KEY, fetch=down))
        result, seen, _ = self.run_pipeline(lookup, TERMS)
        self.assertTrue(result.accepted)
        definitions = result.trace["results"]["reference_definitions"]
        self.assertEqual(definitions["entries"], [])
        reasons = {a["reason"] for u in definitions["unresolved"] for a in u["attempts"]}
        self.assertEqual(reasons, {"api_error", "not_configured"})
        self.assertEqual(seen["Simplifier"]["glossary"], NO_GLOSSARY)


if __name__ == "__main__":
    unittest.main()
