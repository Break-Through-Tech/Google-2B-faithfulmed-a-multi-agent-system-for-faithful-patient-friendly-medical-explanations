"""Reference definitions from UMLS for terms in extracted facts.

Workflow (runs right after the Extractor):

    Extractor lookup_terms --code validation against facts/source evidence-->
                           --UMLS lookups (deterministic code)-->
                           reference definitions

Reference definitions are general background about what a term means. They are
NOT facts about the patient, so they are kept in their own structure and never
merged into the Extractor's patient-fact list. The Extractor only identifies
terms; it never writes or edits a definition. Every definition comes verbatim
(cleaned of markup and citation tags, possibly shortened) from UMLS, with its
CUI. No separate Term Finder model call is made.

Any failure (no credentials, network error, unknown term, bad Extractor term output)
becomes an "unresolved" record or an error note. It never raises into the
pipeline and never produces a substitute definition.

Configuration (environment variables, see .env.example):
    UMLS_API_KEY         UTS API key (free with a UMLS license).
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import DEFAULT_MODEL

REFERENCE_DEFINITIONS_SCHEMA = "faithfulmed_reference_definitions_v4"
CATEGORIES = ("medication", "clinical_concept")
MAX_DEFINITION_CHARS = 600
MAX_CANDIDATES_PER_TERM = 10
NO_GLOSSARY = (
    "NO GLOSSARY ENTRIES ARE AVAILABLE. Do not define, explain, expand, or add "
    "medical background about any term unless that explanation is explicitly "
    "stated in the medical source. Preserve unexplained medical terms as written."
)
REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_NOTICE = (
    "General reference definitions from UMLS. They describe what a term "
    "means in general and are not facts about this patient."
)

# UMLS root sources whose definitions are accepted, in order of preference.
# Prefer consumer-facing MedlinePlus text when it is concise enough to remain a
# definition. A truncated MedlinePlus topic page falls behind a complete concise
# definition rather than contributing an arbitrary opening fragment.
UMLS_DEFINITION_SOURCES = ("MEDLINEPLUS", "MSH", "NCI", "CSP", "HPO")

DRUG_SEMANTIC_TYPES = {
    "Antibiotic", "Clinical Drug", "Drug Delivery Device", "Hormone",
    "Pharmacologic Substance", "Vitamin",
}
PROCEDURE_SEMANTIC_TYPES = {
    "Diagnostic Procedure", "Health Care Activity", "Laboratory Procedure",
    "Therapeutic or Preventive Procedure",
}
MEASUREMENT_SEMANTIC_TYPES = {
    "Laboratory Procedure", "Laboratory or Test Result", "Quantitative Concept",
}
EXPLANATORY_CONCEPT_SEMANTIC_TYPES = {
    "Amino Acid, Peptide, or Protein", "Anatomical Structure",
    "Biologically Active Substance", "Body Part, Organ, or Organ Component",
    "Body Substance", "Chemical", "Disease or Syndrome", "Enzyme",
    "Finding", "Hormone", "Injury or Poisoning", "Mental or Behavioral Dysfunction",
    "Organic Chemical", "Pathologic Function", "Sign or Symptom", "Vitamin",
}
PROCEDURE_TERM_RE = re.compile(
    r"\b(?:biopsy|dialysis|endoscopy|imaging|panel|procedure|scan|surgery|test|therapy|ultrasound)\b",
    re.I,
)
MISLEADING_NAME_RE = re.compile(
    r"\b(?:administration|assay|concentration|measurement|susceptibility)\b|(?:Ser/Plas|OrdQn|MCnc)",
    re.I,
)
GENETIC_SEMANTIC_TYPES = {
    "Amino Acid Sequence", "Gene or Genome", "Nucleotide Sequence",
}
GENETIC_CONTEXT_RE = re.compile(
    r"\b(?:allele|chromosome|dna|gene|genetic|genomic|mutation|rna|variant)\b", re.I,
)
ABBREVIATION_RE = re.compile(r"[A-Za-z0-9]{2,6}")

Fetch = Callable[[str], Optional[Any]]


class LookupServiceError(RuntimeError):
    """A lookup service could not be reached or returned an unusable response."""


def _norm(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _appears_in(term: str, text: str) -> bool:
    """Whole-word, case-insensitive, whitespace-tolerant containment."""
    pattern = r"\s+".join(re.escape(part) for part in term.split())
    return re.search(rf"(?<!\w){pattern}(?!\w)", text, re.I) is not None


def _strip_fences(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, re.S)
    return match.group(1).strip() if match else text.strip()


def parse_term_candidates(
    raw: str | List[Dict[str, Any]],
    facts: List[Dict[str, Any]],
    source_text: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validate Extractor-selected terms against facts or exact source evidence.

    Returns ``(candidates, dropped)``. A term is kept only when it appears in a
    cited fact or in exact source evidence, so a general medical document can
    request definitions without inventing patient facts and the LLM still cannot
    introduce a term that is absent from the source.
    Raises ValueError only when the output is not a JSON list at all.
    """
    if isinstance(raw, list):
        items = raw
    else:
        try:
            items = json.loads(_strip_fences(raw))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Extractor lookup_terms is not valid JSON: {exc}") from exc
    if not isinstance(items, list):
        raise ValueError("Extractor lookup_terms must be a JSON list.")

    fact_text = {f["fact_id"]: f["fact_text"] for f in facts}
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    dropped: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            dropped.append({"item": item, "reason": "not_an_object"})
            continue
        term, category, ids = item.get("term"), item.get("category"), item.get("fact_ids")
        evidence = item.get("source_evidence")
        if not isinstance(term, str) or not term.strip():
            dropped.append({"item": item, "reason": "missing_term"})
            continue
        term = " ".join(term.split())
        if category not in CATEGORIES:
            dropped.append({"term": term, "reason": "invalid_category"})
            continue
        if not isinstance(ids, list):
            ids = []
        if not isinstance(evidence, list):
            evidence = []
        valid_ids = [i for i in ids if isinstance(i, str) and i in fact_text and _appears_in(term, fact_text[i])]
        valid_evidence = [
            text for text in evidence
            if isinstance(text, str) and text and text in source_text and _appears_in(term, text)
        ]
        if not valid_ids and not valid_evidence:
            dropped.append({"term": term, "reason": "term_not_in_cited_source"})
            continue
        key = (_norm(term), category)
        if key in merged:
            merged[key]["fact_ids"].extend(i for i in valid_ids if i not in merged[key]["fact_ids"])
            merged[key]["source_evidence"].extend(
                text for text in valid_evidence if text not in merged[key]["source_evidence"]
            )
        else:
            merged[key] = {
                "term": term,
                "category": category,
                "fact_ids": list(dict.fromkeys(valid_ids)),
                "source_evidence": list(dict.fromkeys(valid_evidence)),
            }
    return list(merged.values()), dropped


# ── Shared result shapes ────────────────────────────────────────────────

def _miss(source: str, reason: str, detail: str = "") -> Dict[str, Any]:
    return {"status": "miss", "source": source, "reason": reason, "detail": detail}


def _clean_definition(text: str, limit: int = MAX_DEFINITION_CHARS) -> Tuple[str, bool]:
    """Remove markup and citation tags; shorten at a sentence boundary if long."""
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    tag = r"(?:[A-Z]{1,3}\d+|FDA Label|Label|Link)"
    text = re.sub(rf"\s*\[{tag}(?:\s*,\s*{tag})*\]", "", text)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind(". "))
    if end >= limit // 4:
        return cut[: end + 1], True
    return cut[: cut.rfind(" ")].rstrip(",;:") + " …", True


# ── UMLS (UTS REST API) ─────────────────────────────────────────────────

def _http_json(url: str, timeout: float) -> Optional[Any]:
    """GET JSON. Returns None on 404. Raises LookupServiceError on other failures."""
    request = urllib.request.Request(url, headers={"Accept": "application/json",
                                                   "User-Agent": "FaithfulMed/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise LookupServiceError(f"HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LookupServiceError(f"network error: {type(exc).__name__}") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise LookupServiceError("response was not JSON") from None


class UMLSClient:
    """Look up a term's UMLS concept (CUI) and a definition from an allowed source."""

    BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
    SOURCE = "UMLS"

    def __init__(self, api_key: str, *, version: str = "current", fetch: Optional[Fetch] = None,
                 timeout: float = 10.0, definition_sources: Tuple[str, ...] = UMLS_DEFINITION_SOURCES):
        if not api_key:
            raise ValueError("UMLSClient needs a UTS API key.")
        self._api_key = api_key
        self.version = version
        self._fetch = fetch or (lambda url: _http_json(url, timeout))
        self.definition_sources = definition_sources
        self._cache: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

    def _get(self, path: str, **params: Any) -> Optional[Any]:
        query = urllib.parse.urlencode({**params, "apiKey": self._api_key})
        try:
            return self._fetch(f"{self.BASE_URL}{path}?{query}")
        except LookupServiceError as exc:
            # Never let the key leak into traces through an error message.
            raise LookupServiceError(str(exc).replace(self._api_key, "***")) from None

    def _search(self, term: str) -> List[Dict[str, Any]]:
        for search_type in ("exact", "normalizedString"):
            data = self._get(f"/search/{self.version}", string=term, searchType=search_type, pageSize=10)
            results = ((data or {}).get("result") or {}).get("results") or []
            hits = [r for r in results if isinstance(r, dict) and r.get("ui") not in (None, "", "NONE")]
            if hits:
                # Bound alternative CUIs per term, not the number of report terms.
                # This prevents one ambiguous string from causing hundreds of
                # definition requests while every Extractor term is still attempted.
                return hits[:MAX_CANDIDATES_PER_TERM]
        return []

    @staticmethod
    def _semantic_types(hit: Dict[str, Any]) -> set[str]:
        """Normalize semantic types returned by current and older UMLS responses."""
        values = hit.get("semanticTypes") or []
        return {
            value.get("name") if isinstance(value, dict) else value
            for value in values
            if (isinstance(value, str) and value) or (
                isinstance(value, dict) and isinstance(value.get("name"), str)
            )
        }

    @staticmethod
    def _context_profile(term: str, category: str, context: str,
                         fact_types: Tuple[str, ...]) -> str:
        if category == "medication":
            return "medication"
        if "procedure" in fact_types or PROCEDURE_TERM_RE.search(term):
            return "procedure"
        if re.search(r"\b(?:underwent|performed|procedure|surgery)\b", context, re.I):
            return "procedure"
        return "explanatory_concept"

    def _hit_score(self, hit: Dict[str, Any], term: str, profile: str) -> int:
        """Rank same-name CUIs by semantic fit for patient-facing explanations."""
        name = str(hit.get("name") or "")
        semantic_types = self._semantic_types(hit)
        score = 0
        if _norm(name) == _norm(term):
            score += 40
        elif _norm(term) in _norm(name) or _norm(name) in _norm(term):
            score += 15

        if profile == "medication":
            if semantic_types & DRUG_SEMANTIC_TYPES:
                score += 120
            if semantic_types & MEASUREMENT_SEMANTIC_TYPES:
                score -= 120
            if semantic_types & (PROCEDURE_SEMANTIC_TYPES - {"Laboratory Procedure"}):
                score -= 60
        elif profile == "procedure":
            if semantic_types & PROCEDURE_SEMANTIC_TYPES:
                score += 100
            if semantic_types & DRUG_SEMANTIC_TYPES:
                score -= 50
        else:
            if semantic_types & EXPLANATORY_CONCEPT_SEMANTIC_TYPES:
                score += 70
            if semantic_types & MEASUREMENT_SEMANTIC_TYPES:
                score -= 100

        if profile != "procedure" and MISLEADING_NAME_RE.search(name):
            score -= 90
        if profile == "procedure" and re.search(r"\b(?:administration|route)\b", name, re.I):
            score -= 90
        return score

    @staticmethod
    def _abbreviation_matches_name(term: str, name: str) -> bool:
        """Reject opaque short-code matches unless the candidate name fits it."""
        compact = re.sub(r"[^A-Za-z0-9]", "", term)
        if not ABBREVIATION_RE.fullmatch(compact) or sum(c.isupper() for c in term) < 2:
            return True
        acronym = compact[:-1] if compact.endswith("s") and compact[-2:-1].isupper() else compact
        words = re.findall(r"[A-Za-z0-9]+", name)
        initials = "".join(word[0] for word in words if word.lower() not in {"a", "an", "and", "of", "the"})
        return _norm(term) == _norm(name) or acronym.upper() == initials.upper()

    def _hit_is_compatible(self, hit: Dict[str, Any], term: str, profile: str, context: str) -> bool:
        """Fail closed when UMLS returns a concept incompatible with source context."""
        name = str(hit.get("name") or "")
        semantic_types = self._semantic_types(hit)
        if not self._abbreviation_matches_name(term, name):
            return False
        if semantic_types & GENETIC_SEMANTIC_TYPES and not GENETIC_CONTEXT_RE.search(context):
            return False
        if profile == "medication" and semantic_types and not semantic_types & DRUG_SEMANTIC_TYPES:
            return False
        if profile == "procedure" and semantic_types and not semantic_types & PROCEDURE_SEMANTIC_TYPES:
            return False
        if profile == "explanatory_concept" and self._hit_score(hit, term, profile) < 0:
            return False

        lowered_context, lowered_name = context.lower(), name.lower()
        renal_context = re.search(r"\b(?:kidney|renal|urinary|urine|ureter|bladder)\b", lowered_context)
        brain_name = re.search(r"\b(?:brain|cerebral|neural|neurologic)\b", lowered_name)
        brain_context = re.search(r"\b(?:brain|cerebral|neural|neurologic)\b", lowered_context)
        renal_name = re.search(r"\b(?:kidney|renal|urinary|ureter|bladder)\b", lowered_name)
        if (renal_context and brain_name) or (brain_context and renal_name):
            return False
        return True

    def _rank_hits(self, hits: List[Dict[str, Any]], term: str, profile: str) -> List[Dict[str, Any]]:
        """Sort by semantic/context fit while preserving UMLS order for ties."""
        indexed = list(enumerate(hits))
        indexed.sort(key=lambda pair: (-self._hit_score(pair[1], term, profile), pair[0]))
        return [hit for _, hit in indexed]

    def lookup(self, term: str, *, category: str = "clinical_concept", context: str = "",
               fact_types: Tuple[str, ...] = ()) -> Dict[str, Any]:
        profile = self._context_profile(term, category, context, fact_types)
        key = (_norm(term), category, profile, _norm(context))
        if key not in self._cache:
            self._cache[key] = self._lookup(term, profile, context)
        return dict(self._cache[key])

    def _lookup(self, term: str, profile: str, context: str) -> Dict[str, Any]:
        hits = self._search(term)
        if not hits:
            return _miss(self.SOURCE, "not_found", "No UMLS concept matches this term exactly.")
        ranked = self._rank_hits(hits, term, profile)
        compatible = [hit for hit in ranked if self._hit_is_compatible(hit, term, profile, context)]
        if not compatible:
            names = ", ".join(f"{hit.get('ui')} ({hit.get('name', '')})" for hit in ranked[:5])
            return _miss(
                self.SOURCE, "context_mismatch",
                f"UMLS candidates were incompatible with the term category or source context: {names}.",
            )
        cuis = list(dict.fromkeys(h["ui"] for h in compatible))
        # A short all-caps abbreviation that maps to several concepts (MS, PE, RA...)
        # cannot be resolved safely without context, so it is left unresolved.
        if len(cuis) > 1 and re.fullmatch(r"[A-Z0-9]{2,6}", term.strip()):
            return _miss(self.SOURCE, "ambiguous",
                         f"Abbreviation matches {len(cuis)} UMLS concepts: {', '.join(cuis[:5])}.")
        for best in compatible:
            cui = best["ui"]
            data = self._get(f"/content/{self.version}/CUI/{cui}/definitions", pageSize=100)
            definitions = (data or {}).get("result") or []
            by_source: Dict[str, str] = {}
            for d in definitions:
                if isinstance(d, dict) and isinstance(d.get("value"), str) and d["value"].strip():
                    by_source.setdefault(d.get("rootSource", ""), d["value"])
            cleaned = {
                source: _clean_definition(value)
                for source, value in by_source.items()
                if source in self.definition_sources
            }
            has_complete_alternative = any(
                definition and not truncated and source != "MEDLINEPLUS"
                for source, (definition, truncated) in cleaned.items()
            )
            chosen = next((
                source for source in self.definition_sources
                if source in cleaned and cleaned[source][0]
                and not (source == "MEDLINEPLUS" and cleaned[source][1] and has_complete_alternative)
            ), None)
            if chosen is None:
                continue
            definition, truncated = cleaned[chosen]
            return {
                "status": "found", "source": self.SOURCE, "source_id": cui,
                "matched_name": best.get("name") or "", "definition": definition,
                "definition_source": chosen, "source_version": self.version,
                "semantic_types": sorted(self._semantic_types(best)),
                "selection_profile": profile,
                "reference_url": f"https://uts.nlm.nih.gov/uts/umls/concept/{cui}", "truncated": truncated,
            }
        names = ", ".join(
            f"{hit.get('ui')} ({hit.get('name', '')})" for hit in compatible[:5]
        )
        return _miss(
            self.SOURCE, "no_definition",
            f"No context-compatible UMLS candidate had a definition from an allowed source. Tried: {names}.",
        )


# ── Lookup service ──────────────────────────────────────────────────────

def lookup_config_from_env(env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Cheap description of which sources are configured (for cache keys and traces)."""
    env = os.environ if env is None else env
    return {"umls": bool(env.get("UMLS_API_KEY"))}


class MedicalLookup:
    """Routes each term to the right source and records every hit and miss."""

    def __init__(self, umls: Optional[UMLSClient] = None, *,
                 workers: int = 4, config_errors: Optional[List[str]] = None):
        self.umls = umls
        self.workers = workers
        self.config_errors = list(config_errors or [])

    @classmethod
    def disabled(cls) -> "MedicalLookup":
        return cls()

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "MedicalLookup":
        env = os.environ if env is None else env
        key = env.get("UMLS_API_KEY", "").strip()
        return cls(UMLSClient(key, version=env.get("UMLS_VERSION") or "current") if key else None)

    @property
    def configured(self) -> bool:
        return self.umls is not None

    def sources(self) -> Dict[str, Any]:
        return {
            "umls": {"configured": self.umls is not None,
                     "version": self.umls.version if self.umls else None,
                     "definition_sources": list(self.umls.definition_sources) if self.umls else []},
        }

    def _try(self, client: Any, source: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
        if client is None:
            return _miss(source, "not_configured", f"{source} lookup is not configured.")
        try:
            return client.lookup(
                candidate["term"], category=candidate["category"],
                context=candidate.get("context", ""),
                fact_types=tuple(candidate.get("fact_types") or ()),
            )
        except Exception as exc:  # any service failure becomes a recorded miss
            return _miss(source, "api_error", f"{type(exc).__name__}: {exc}")

    def lookup_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        term = candidate["term"]
        order = [(self.umls, UMLSClient.SOURCE)]  # medications and clinical concepts alike
        attempts = []
        for client, source in order:
            result = self._try(client, source, candidate)
            if result["status"] == "found":
                found = {k: v for k, v in result.items() if k != "status"}
                return {"found": True, "entry": {"term": term, "category": candidate["category"],
                                                 "fact_ids": candidate["fact_ids"],
                                                 "source_evidence": candidate.get("source_evidence", []),
                                                 "evidence_type": "reference_definition",
                                                 "allowed_use": "general_term_explanation_only",
                                                 **found}}
            attempts.append({k: v for k, v in result.items() if k != "status"})
        return {"found": False, "unresolved": {"term": term, "category": candidate["category"],
                                               "fact_ids": candidate["fact_ids"],
                                               "source_evidence": candidate.get("source_evidence", []),
                                               "attempts": attempts}}

    def lookup_candidates(self, candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool:
            results = list(pool.map(self.lookup_candidate, candidates))
        entries = [r["entry"] for r in results if r["found"]]
        unresolved = [r["unresolved"] for r in results if not r["found"]]
        return entries, unresolved


def empty_reference_definitions(status: str, lookup: Optional[MedicalLookup] = None,
                                errors: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "schema": REFERENCE_DEFINITIONS_SCHEMA, "status": status, "notice": REFERENCE_NOTICE,
        "sources": lookup.sources() if lookup else {}, "entries": [], "unresolved": [],
        "dropped_terms": [], "extractor_lookup_terms": [],
        "errors": list(errors or []) + (lookup.config_errors if lookup else []),
    }


async def build_reference_definitions(
    facts: List[Dict[str, Any]],
    lookup_terms: List[Dict[str, Any]],
    lookup: MedicalLookup,
    *,
    source_text: str = "",
) -> Dict[str, Any]:
    """Validate Extractor terms and look them up without another model call.

    ``status`` is "completed" (lookups ran; some terms may be unresolved),
    "not_configured" (no source configured), or "failed" (the Extractor term
    selection was unusable or lookup failed).
    """
    if not lookup.configured:
        result = empty_reference_definitions("not_configured", lookup)
        result["extractor_lookup_terms"] = lookup_terms
        return result
    result = empty_reference_definitions("completed", lookup)
    result["extractor_lookup_terms"] = lookup_terms
    try:
        candidates, dropped = parse_term_candidates(lookup_terms, facts, source_text)
    except Exception as exc:
        result.update(status="failed")
        result["errors"].append(f"Extractor term validation failed: {type(exc).__name__}: {exc}")
        return result
    result["dropped_terms"] = dropped
    fact_by_id = {fact["fact_id"]: fact for fact in facts}
    candidates_with_context = []
    for candidate in candidates:
        cited = [fact_by_id[fact_id] for fact_id in candidate["fact_ids"] if fact_id in fact_by_id]
        evidence = candidate.get("source_evidence") or []
        term_windows = []
        if source_text:
            for match in re.finditer(re.escape(candidate["term"]), source_text, re.I):
                term_windows.append(source_text[max(0, match.start() - 300):match.end() + 300])
                if len(term_windows) == 3:
                    break
        candidates_with_context.append({
            **candidate,
            "context": " ".join([
                *(str(fact.get("fact_text") or "") for fact in cited),
                *(
                    str(text)
                    for fact in cited
                    for text in (fact.get("source_evidence") or [])
                ),
                *evidence,
                *term_windows,
            ]),
            "fact_types": list(dict.fromkeys(str(fact.get("fact_type") or "") for fact in cited)),
        })
    try:
        entries, unresolved = await asyncio.to_thread(lookup.lookup_candidates, candidates_with_context)
    except Exception as exc:  # defensive: lookup_candidate already traps service errors
        result.update(status="failed")
        result["errors"].append(f"Lookup failed: {type(exc).__name__}: {exc}")
        return result
    result["entries"], result["unresolved"] = entries, unresolved
    return result


async def run_extractor_with_lookups(
    source_text: str, model: str = DEFAULT_MODEL, *, lookup: Optional[MedicalLookup] = None,
) -> Dict[str, Any]:
    """Run one Extractor call, then attach Python-fetched UMLS definitions.

    Definitions remain a separate, provenance-labeled field rather than being
    mixed into patient facts.
    """
    from .extractor import parse_extractor_output, patient_facts_json, run_extractor

    raw = await run_extractor(source_text, model)
    lookup = lookup if lookup is not None else MedicalLookup.from_env()
    try:
        extracted = parse_extractor_output(raw)
    except ValueError as exc:
        return {"extractor_output": raw, "facts": raw, "reference_definitions": empty_reference_definitions(
            "failed", lookup, [f"Extractor output is invalid, so no lookups ran: {exc}"])}
    facts = [fact.model_dump() for fact in extracted.patient_facts]
    terms = [term.model_dump() for term in extracted.lookup_terms]
    definitions = await build_reference_definitions(facts, terms, lookup, source_text=source_text)
    return {
        "extractor_output": extracted.model_dump(),
        "facts": patient_facts_json(extracted),
        "reference_definitions": definitions,
    }


# ── Formats handed to the downstream agents ────────────────────────────

def render_glossary(definitions: Dict[str, Any]) -> str:
    """Plain-text glossary for the Simplifier's {glossary} slot."""
    entries = (definitions or {}).get("entries") or []
    if not entries:
        return NO_GLOSSARY
    lines = [
        REFERENCE_NOTICE,
        "Closed list: do not explain any medical term absent from this list unless the medical source explicitly explains it.",
        "Allowed use: explain what the matching term means in general.",
        "Forbidden use: infer this patient's diagnosis, current status, symptoms, cause, severity, treatment indication, or instructions.",
        "",
    ]
    for e in entries:
        ids = ", ".join(e.get("fact_ids") or [])
        grounding = f"for facts {ids}" if ids else "from exact source evidence"
        lines.append(f"- {e['term']} [{e['source']} {e['source_id']}; matched \"{e['matched_name']}\"; "
                     f"definition from {e['definition_source']}; {grounding}]: {e['definition']}")
    return "\n".join(lines)


def reference_definitions_for_prompt(definitions: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compact entries for the Verifier's input JSON."""
    keep = (
        "term", "fact_ids", "source_evidence", "source", "source_id", "matched_name",
        "definition_source", "definition", "semantic_types", "selection_profile",
        "evidence_type", "allowed_use",
    )
    return [{k: e.get(k) for k in keep} for e in (definitions or {}).get("entries") or []]


# ── CLI ─────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    look = sub.add_parser("lookup", help="Look up one term (UMLS calls use the network).")
    look.add_argument("term")
    look.add_argument("--category", choices=CATEGORIES, default="clinical_concept")
    args = parser.parse_args(argv)

    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.replace("export ", "").strip(), value.strip().strip("'\""))
    lookup = MedicalLookup.from_env()
    if not lookup.configured:
        print("No lookup source configured. Set UMLS_API_KEY.", file=sys.stderr)
        return 1
    result = lookup.lookup_candidate({"term": args.term, "category": args.category, "fact_ids": []})
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
