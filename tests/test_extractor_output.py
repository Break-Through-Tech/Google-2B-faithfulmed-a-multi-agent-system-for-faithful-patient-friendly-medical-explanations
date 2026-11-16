"""Extractor output parsing and narrowly scoped recovery tests."""

from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from agents.extractor import parse_extractor_output, patient_facts_json


def output_with_fact(fact: dict) -> str:
    return json.dumps({"patient_facts": [fact], "lookup_terms": []})


BASE_FACT = {
    "fact_id": "F-001",
    "fact_text": "Discussed condition and treatment with the patient.",
    "fact_type": "instruction",
    "source_evidence": ["Discussed condition and treatment with the patient."],
}


class ExtractorOutputRecoveryTests(unittest.TestCase):
    def test_redundant_fact_type_marker_is_removed(self) -> None:
        fact = {**BASE_FACT, "instruction": "instruction"}
        parsed = parse_extractor_output(output_with_fact(fact))
        self.assertEqual(parsed.patient_facts[0].fact_type, "instruction")
        self.assertNotIn("instruction", json.loads(patient_facts_json(parsed))[0])

    def test_nonredundant_type_named_field_is_still_rejected(self) -> None:
        fact = {**BASE_FACT, "instruction": "Take a new medication."}
        with self.assertRaises(ValidationError):
            parse_extractor_output(output_with_fact(fact))

    def test_arbitrary_extra_field_is_still_rejected(self) -> None:
        fact = {**BASE_FACT, "unsupported_extra": "value"}
        with self.assertRaises(ValidationError):
            parse_extractor_output(output_with_fact(fact))

    def test_general_document_term_can_use_source_evidence_without_patient_fact(self) -> None:
        source = "Gestational diabetes occurs during pregnancy."
        parsed = parse_extractor_output(json.dumps({
            "patient_facts": [],
            "lookup_terms": [{
                "term": "Gestational diabetes",
                "category": "clinical_concept",
                "fact_ids": [],
                "source_evidence": [source],
            }],
        }))
        self.assertEqual(parsed.patient_facts, [])
        self.assertEqual(parsed.lookup_terms[0].source_evidence, [source])

    def test_lookup_term_requires_a_grounding_path(self) -> None:
        with self.assertRaises(ValidationError):
            parse_extractor_output(json.dumps({
                "patient_facts": [],
                "lookup_terms": [{
                    "term": "diabetes", "category": "clinical_concept",
                    "fact_ids": [], "source_evidence": [],
                }],
            }))


if __name__ == "__main__":
    unittest.main()
