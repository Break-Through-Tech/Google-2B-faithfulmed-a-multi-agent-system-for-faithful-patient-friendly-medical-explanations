"""Offline tests for UMLS reference definitions. No network, no Gemini."""

from __future__ import annotations

import asyncio
import json
import unittest
import urllib.parse
from types import SimpleNamespace
from unittest.mock import patch

from agents.extractor import ExtractorOutput
from agents.medical_lookup import (
    NO_GLOSSARY, REFERENCE_DEFINITIONS_SCHEMA,
    LookupServiceError, MedicalLookup, UMLSClient, build_reference_definitions,
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



def fake_umls(concepts: dict, definitions: dict, calls: list | None = None):
    """Concept values are (cui, name) or (cui, name, semantic_types)."""
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
            results = []
            for hit in hits:
                cui, name, *semantic_types = hit
                results.append({
                    "ui": cui, "name": name,
                    "semanticTypes": semantic_types[0] if semantic_types else [],
                })
            return {"result": {"results": results}}
        cui = parsed.path.split("/CUI/")[1].split("/")[0]
        if cui not in definitions:
            return None  # UTS answers 404 when a concept has no definitions
        return {"result": [{"rootSource": s, "value": v} for s, v in definitions[cui]]}
    return fetch


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

    def test_concise_medlineplus_definition_is_preferred(self) -> None:
        client = self.client(
            concepts={"hypertension": [("C0020538", "Hypertensive disease")]},
            definitions={"C0020538": [
                ("MSH", "Persistently high systemic arterial blood pressure."),
                ("MEDLINEPLUS", "High blood pressure that stays elevated over time."),
            ]},
        )
        result = client.lookup("hypertension")
        self.assertEqual(result["definition_source"], "MEDLINEPLUS")
        self.assertEqual(result["definition"], "High blood pressure that stays elevated over time.")

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

    def test_medication_prefers_drug_over_lab_measurement(self) -> None:
        client = self.client(
            concepts={"lidocaine": [
                ("C-MEASURE", "Lidocaine measurement", ["Laboratory Procedure"]),
                ("C-DRUG", "Lidocaine", ["Pharmacologic Substance"]),
            ]},
            definitions={
                "C-MEASURE": [("MSH", "A quantitative lidocaine measurement.")],
                "C-DRUG": [("NCI", "A local anesthetic medication.")],
            },
        )
        result = client.lookup(
            "lidocaine", category="medication", context="Lidocaine was injected.",
            fact_types=("medication",),
        )
        self.assertEqual(result["source_id"], "C-DRUG")
        self.assertEqual(result["selection_profile"], "medication")
        self.assertEqual(result["semantic_types"], ["Pharmacologic Substance"])

    def test_plain_analyte_prefers_substance_over_measurement(self) -> None:
        client = self.client(
            concepts={"creatinine": [
                ("C-MEASURE", "Creatinine measurement", ["Laboratory Procedure"]),
                ("C-SUBSTANCE", "Creatinine", ["Organic Chemical"]),
            ]},
            definitions={
                "C-MEASURE": [("MSH", "A quantitative measurement in a sample.")],
                "C-SUBSTANCE": [("MSH", "A waste product filtered by the kidneys.")],
            },
        )
        result = client.lookup(
            "creatinine", context="The creatinine level was elevated.",
            fact_types=("lab_value",),
        )
        self.assertEqual(result["source_id"], "C-SUBSTANCE")
        self.assertEqual(result["selection_profile"], "explanatory_concept")

    def test_procedure_prefers_therapy_over_administration_concept(self) -> None:
        client = self.client(
            concepts={"dialysis": [
                ("C-ADMIN", "Dialysis method of administration", ["Functional Concept"]),
                ("C-THERAPY", "Renal dialysis", ["Therapeutic or Preventive Procedure"]),
            ]},
            definitions={
                "C-ADMIN": [("MSH", "A method of drug administration.")],
                "C-THERAPY": [("MSH", "A treatment that filters waste from the blood.")],
            },
        )
        result = client.lookup(
            "dialysis", context="The patient receives dialysis.",
            fact_types=("procedure",),
        )
        self.assertEqual(result["source_id"], "C-THERAPY")
        self.assertEqual(result["selection_profile"], "procedure")

    def test_ranked_lookup_tries_next_candidate_when_first_has_no_definition(self) -> None:
        client = self.client(
            concepts={"amoxicillin": [
                ("C-DRUG-NO-DEF", "Amoxicillin", ["Pharmacologic Substance"]),
                ("C-DRUG-WITH-DEF", "Amoxicillin drug", ["Clinical Drug"]),
            ]},
            definitions={"C-DRUG-WITH-DEF": [("NCI", "An antibacterial medication.")]},
        )
        result = client.lookup("amoxicillin", category="medication")
        self.assertEqual(result["source_id"], "C-DRUG-WITH-DEF")

    def test_opaque_abbreviation_rejects_unrelated_gene(self) -> None:
        client = self.client(
            concepts={"tms": [("C-GENE", "PYCARD wt Allele", ["Gene or Genome"])]},
            definitions={"C-GENE": [("NCI", "A human gene allele.")]},
        )
        result = client.lookup("TMs", context="TMs were clear.")
        self.assertEqual(result["reason"], "context_mismatch")

    def test_anatomical_context_rejects_cerebral_match_for_kidney_report(self) -> None:
        client = self.client(
            concepts={"cortical thinning": [
                ("C-BRAIN", "Cerebral Cortical Thinning", ["Finding"]),
            ]},
            definitions={"C-BRAIN": [("HPO", "Reduced thickness of the cerebral cortex.")]},
        )
        result = client.lookup(
            "cortical thinning",
            context="Renal ultrasound of the kidneys showed no cortical thinning.",
        )
        self.assertEqual(result["reason"], "context_mismatch")


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
            {"term": "metoprolol", "category": "medication", "fact_ids": ["F-001"],
             "source_evidence": []},
            {"term": "Hypertension", "category": "clinical_concept", "fact_ids": ["F-001", "F-002"],
             "source_evidence": []},
        ])
        reasons = {d.get("term"): d["reason"] for d in dropped}
        self.assertEqual(reasons["high blood pressure"], "term_not_in_cited_source")
        self.assertEqual(reasons["metop"], "term_not_in_cited_source")  # no partial-word matches
        self.assertEqual(reasons["aspirin"], "invalid_category")

    def test_invalid_output_raises(self) -> None:
        for raw in ("not json", '{"term": "x"}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_term_candidates(raw, FACTS)

    def test_list_input_from_extractor_is_accepted_directly(self) -> None:
        candidates, dropped = parse_term_candidates([
            {"term": "metoprolol", "category": "medication", "fact_ids": ["F-001"]},
        ], FACTS)
        self.assertEqual(candidates[0]["term"], "metoprolol")
        self.assertEqual(dropped, [])

    def test_source_only_term_is_valid_without_patient_facts(self) -> None:
        source = "Gestational diabetes is discussed during pregnancy."
        candidates, dropped = parse_term_candidates([{
            "term": "Gestational diabetes",
            "category": "clinical_concept",
            "fact_ids": [],
            "source_evidence": [source],
        }], [], source)
        self.assertEqual(dropped, [])
        self.assertEqual(candidates, [{
            "term": "Gestational diabetes",
            "category": "clinical_concept",
            "fact_ids": [],
            "source_evidence": [source],
        }])

    def test_source_only_term_requires_exact_evidence(self) -> None:
        candidates, dropped = parse_term_candidates([{
            "term": "diabetes", "category": "clinical_concept", "fact_ids": [],
            "source_evidence": ["Paraphrased evidence about diabetes."],
        }], [], "The source mentions diabetes.")
        self.assertEqual(candidates, [])
        self.assertEqual(dropped[0]["reason"], "term_not_in_cited_source")


def configured_lookup() -> MedicalLookup:
    return MedicalLookup(umls=UMLSClient(KEY, fetch=fake_umls(
        {"hypertension": [("C0020538", "Hypertensive disease")], "metoprolol": [("C0025859", "Metoprolol")]},
        {"C0020538": [("MSH", "Persistently high systemic arterial blood pressure.")],
         "C0025859": [("NCI", "A beta blocker.")]},
    )))


TERMS = [
    {"term": "metoprolol", "category": "medication", "fact_ids": ["F-001"]},
    {"term": "hypertension", "category": "clinical_concept", "fact_ids": ["F-001", "F-002"]},
    {"term": "b.i.d", "category": "clinical_concept", "fact_ids": ["F-001"]},
]


class BuildReferenceDefinitionsTests(unittest.TestCase):
    def run_build(self, lookup, terms=TERMS):
        return asyncio.run(build_reference_definitions(FACTS, terms, lookup))

    def test_successful_lookup_routes_to_the_right_source(self) -> None:
        result = self.run_build(configured_lookup())
        self.assertEqual(result["schema"], REFERENCE_DEFINITIONS_SCHEMA)
        self.assertEqual(result["status"], "completed")
        by_term = {e["term"]: e for e in result["entries"]}
        self.assertEqual((by_term["metoprolol"]["source"], by_term["metoprolol"]["source_id"]), ("UMLS", "C0025859"))
        self.assertEqual((by_term["hypertension"]["source"], by_term["hypertension"]["source_id"]), ("UMLS", "C0020538"))
        for entry in result["entries"]:
            for field in ("term", "definition", "source", "source_id", "reference_url", "fact_ids"):
                self.assertTrue(entry[field], field)
        self.assertEqual([u["term"] for u in result["unresolved"]], ["b.i.d"])
        self.assertEqual(result["unresolved"][0]["attempts"][0]["reason"], "not_found")

    def test_medication_is_looked_up_in_umls(self) -> None:
        umls = UMLSClient(KEY, fetch=fake_umls({"metoprolol": [("C0025859", "Metoprolol")]},
                                               {"C0025859": [("NCI", "A beta blocker.")]}))
        result = MedicalLookup(umls=umls).lookup_candidate(
            {"term": "metoprolol", "category": "medication", "fact_ids": ["F-001"]})
        self.assertTrue(result["found"])
        self.assertEqual(result["entry"]["source"], "UMLS")

    def test_not_configured_skips_umls_calls(self) -> None:
        result = self.run_build(MedicalLookup.disabled())
        self.assertEqual(result["status"], "not_configured")
        self.assertEqual(result["entries"], [])
        self.assertEqual(result["extractor_lookup_terms"], TERMS)
        self.assertEqual(render_glossary(result), NO_GLOSSARY)

    def test_invalid_extractor_term_is_dropped_without_lookup(self) -> None:
        result = self.run_build(configured_lookup(), [
            {"term": "diabetes", "category": "clinical_concept", "fact_ids": ["F-999"]},
        ])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["entries"], [])
        self.assertEqual(result["dropped_terms"][0]["reason"], "term_not_in_cited_source")

    def test_general_document_term_is_looked_up_without_patient_facts(self) -> None:
        source = "Gestational diabetes is discussed during pregnancy."
        terms = [{
            "term": "Gestational diabetes", "category": "clinical_concept",
            "fact_ids": [], "source_evidence": [source],
        }]
        lookup = MedicalLookup(umls=UMLSClient(KEY, fetch=fake_umls(
            {"gestational diabetes": [
                ("C0085207", "Gestational Diabetes", ["Disease or Syndrome"]),
            ]},
            {"C0085207": [("MEDLINEPLUS", "Diabetes that develops during pregnancy.")]},
        )))
        result = asyncio.run(build_reference_definitions(
            [], terms, lookup, source_text=source,
        ))
        self.assertEqual(result["unresolved"], [])
        self.assertEqual(result["entries"][0]["source_id"], "C0085207")
        self.assertEqual(result["entries"][0]["fact_ids"], [])
        self.assertEqual(result["entries"][0]["source_evidence"], [source])

    def test_all_valid_terms_are_looked_up_without_silent_limit(self) -> None:
        facts = [
            {"fact_id": f"F-{index:03d}", "fact_text": f"Term{index} is documented.",
             "fact_type": "diagnosis", "source_evidence": [f"Term{index} is documented."]}
            for index in range(1, 31)
        ]
        terms = [
            {"term": f"Term{index}", "category": "clinical_concept",
             "fact_ids": [f"F-{index:03d}"]}
            for index in range(1, 31)
        ]
        concepts = {
            f"term{index}": [(f"C{index}", f"Term{index}", ["Disease or Syndrome"])]
            for index in range(1, 31)
        }
        definitions = {f"C{index}": [("MSH", f"Definition {index}.")] for index in range(1, 31)}
        lookup = MedicalLookup(umls=UMLSClient(KEY, fetch=fake_umls(concepts, definitions)))
        result = asyncio.run(build_reference_definitions(facts, terms, lookup))
        self.assertEqual(len(result["entries"]), 30)
        self.assertEqual(result["unresolved"], [])
        self.assertFalse(any(item.get("reason") == "term_limit" for item in result["dropped_terms"]))

    def test_env_config(self) -> None:
        env = {"UMLS_API_KEY": "", "DRUGBANK_INDEX_PATH": "/any/index.json"}  # DrugBank is ignored
        self.assertFalse(MedicalLookup.from_env(env).configured)
        self.assertEqual(lookup_config_from_env(env), {"umls": False})
        lookup = MedicalLookup.from_env({"UMLS_API_KEY": KEY})
        self.assertTrue(lookup.configured)
        self.assertEqual(set(lookup.sources()), {"umls"})


class DownstreamFormatTests(unittest.TestCase):
    def definitions(self):
        return asyncio.run(build_reference_definitions(FACTS, TERMS, configured_lookup()))

    def test_glossary_carries_definition_and_provenance(self) -> None:
        glossary = render_glossary(self.definitions())
        self.assertIn("not facts about this patient", glossary)
        self.assertIn("Forbidden use", glossary)
        self.assertIn("metoprolol [UMLS C0025859", glossary)
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
        self.assertEqual({d["source_id"] for d in payload["reference_definitions"]}, {"C0025859", "C0020538"})
        self.assertTrue(all(
            item["allowed_use"] == "general_term_explanation_only"
            for item in payload["reference_definitions"]
        ))
        empty = production_verifier_prompt(SOURCE, json.dumps(FACTS), "Draft", "[]")
        self.assertNotIn("reference_definitions", json.loads(empty.rsplit("INPUT JSON:\n", 1)[1]))


ACCEPT = json.dumps({"overall_decision": "accept", "explanation_accuracy": {"prediction": "yes", "reason": "ok"},
                     "sentences": [], "unsupported_claims": [], "omissions": []})


def stub(name):
    return SimpleNamespace(name=name)


@patch("agents.production_pipeline.create_readability_agent", return_value=stub("Readability"))
@patch("agents.production_pipeline.create_simplifier_agent", return_value=stub("Simplifier"))
@patch("agents.production_pipeline.create_extractor_agent", return_value=stub("Extractor"))
class PipelineIntegrationTests(unittest.TestCase):
    def run_pipeline(self, lookup, terms=TERMS):
        seen: dict = {}
        extracted = json.dumps({"patient_facts": FACTS, "lookup_terms": terms})

        async def executor(agent, state, message, on_text):
            seen[agent.name] = state
            text = {"Extractor": extracted, "Simplifier": "Draft.",
                    "Readability": "Final."}[agent.name]
            on_text(text, text)
            return text

        async def verifier_runner(source, facts, draft, model, on_text, definitions):
            seen["Verifier"] = {"facts": facts, "definitions": json.loads(definitions)}
            return ACCEPT

        result = asyncio.run(run_production_pipeline(
            SOURCE, agent_executor=executor, verifier_runner=verifier_runner, lookup=lookup))
        return result, seen, extracted

    def test_definitions_reach_simplifier_and_verifier_and_facts_are_unchanged(self, *_):
        result, seen, extracted = self.run_pipeline(configured_lookup())
        self.assertTrue(result.accepted)
        self.assertEqual(result.trace["results"]["extractor"], extracted)
        ExtractorOutput.model_validate_json(result.trace["results"]["extractor"])
        self.assertEqual(json.loads(seen["Verifier"]["facts"]), FACTS)  # no definitions mixed into facts
        self.assertEqual(json.loads(seen["Simplifier"]["atoms"]), FACTS)
        self.assertIn("UMLS C0025859", seen["Simplifier"]["glossary"])
        self.assertEqual({d["source_id"] for d in seen["Verifier"]["definitions"]}, {"C0025859", "C0020538"})
        self.assertNotIn("TermFinder", seen)
        enriched = result.trace["results"]["extractor_enriched"]
        self.assertEqual(enriched["patient_facts"], FACTS)
        self.assertEqual(enriched["lookup_terms"], [
            {**term, "source_evidence": []} for term in TERMS
        ])
        stage = next(s for s in result.trace["stages"] if s["agent"] == "UMLSLookup")
        self.assertEqual(stage["status"], "completed")
        self.assertEqual(result.trace["results"]["reference_definitions"]["status"], "completed")

    def test_invalid_extractor_term_does_not_interrupt_pipeline(self, *_):
        result, seen, _ = self.run_pipeline(configured_lookup(), [
            {"term": "not present", "category": "clinical_concept", "fact_ids": ["F-001"]},
        ])
        self.assertTrue(result.accepted)
        self.assertEqual(result.final_explanation, "Final.")
        self.assertEqual(seen["Simplifier"]["glossary"], NO_GLOSSARY)
        self.assertEqual(seen["Verifier"]["definitions"], [])
        stage = next(s for s in result.trace["stages"] if s["agent"] == "UMLSLookup")
        self.assertEqual(stage["status"], "completed")

    def test_api_errors_become_unresolved_terms(self, *_):
        def down(url):
            raise LookupServiceError("HTTP 503")
        lookup = MedicalLookup(umls=UMLSClient(KEY, fetch=down))
        result, seen, _ = self.run_pipeline(lookup)
        self.assertTrue(result.accepted)
        definitions = result.trace["results"]["reference_definitions"]
        self.assertEqual(definitions["entries"], [])
        reasons = {a["reason"] for u in definitions["unresolved"] for a in u["attempts"]}
        self.assertEqual(reasons, {"api_error"})  # UMLS is the only source
        self.assertEqual(seen["Simplifier"]["glossary"], NO_GLOSSARY)


if __name__ == "__main__":
    unittest.main()
