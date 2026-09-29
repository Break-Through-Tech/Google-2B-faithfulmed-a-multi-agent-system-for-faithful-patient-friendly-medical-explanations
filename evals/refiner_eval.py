"""
FaithfulMed — Refiner evaluation harness.

The Refiner agent (agents/refiner.py) has a narrow, checkable contract:
  1. Remove the unsupported claims the Verifier flagged.
  2. Restore the omissions the Verifier flagged.
  3. Change nothing else (minimal, targeted edits).
  4. If the Verifier says the draft is faithful, return the draft unchanged.

This module scores whether a refined draft honored that contract. It is fully
offline and standard-library only: it compares the refined text against the
original draft and the Verifier feedback. It never imports ADK or calls Gemini,
so results can be recomputed cheaply and tests stay tiny.

Provides:
  - parse_verdict(verdict): tolerant reader for Verifier feedback (dict or JSON string)
  - is_unchanged(draft, refined): whether the Refiner left the draft byte-for-byte alone
  - minimal_edit_ratio(draft, refined): 0..1 similarity of refined to draft (1.0 = identical)
  - unsupported_removed_rate(draft, refined, unsupported_claims): fraction of flagged claims removed
  - omissions_restored_rate(refined, omissions): fraction of flagged omissions now present
  - evaluate_single(draft, refined, verdict): dict of all Refiner metrics for one example
  - evaluate_multiple(examples): list of per-example metric dicts
  - summarize_results(results): means and pass rates across many examples
  - compare_texts(draft, refined): before/after edit summary for one example
  - contract_pass_rate(results): percentage of examples that honored the full contract
  - find_failures(results): the examples that broke the contract, with a reason
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

# A "content word" for fuzzy phrase matching: alphanumeric runs, lowercased.
_WORD_RE = re.compile(r"[a-z0-9]+")

# Fraction of a flagged phrase's content words that must be present in the
# refined text for the phrase to count as "still present" / "restored".
DEFAULT_MATCH_THRESHOLD = 0.8


def _tokens(text: str) -> list[str]:
    """Lowercased alphanumeric tokens used for order-independent phrase matching."""
    return _WORD_RE.findall(text.lower())


def contains_phrase(haystack: str, needle: str, threshold: float = DEFAULT_MATCH_THRESHOLD) -> bool:
    """
    Fuzzy containment: True when at least `threshold` fraction of the needle's
    content words appear in the haystack. This tolerates the small rewordings a
    Refiner may make while still detecting whether a claim or fact is present.
    An empty needle is treated as trivially present.
    """
    needle_tokens = _tokens(needle)
    if not needle_tokens:
        return True
    haystack_tokens = set(_tokens(haystack))
    present = sum(1 for token in needle_tokens if token in haystack_tokens)
    return (present / len(needle_tokens)) >= threshold


def parse_verdict(verdict: Any) -> dict:
    """
    Read Verifier feedback into a predictable dict without inventing fields.

    Accepts either a dict (already parsed) or a JSON string (as run_verifier
    returns). Missing or malformed feedback yields empty lists and faithful=None
    so scoring can treat it as "no flags" rather than crashing.
    """
    data: Any = verdict
    if isinstance(verdict, str):
        try:
            data = json.loads(verdict)
        except (json.JSONDecodeError, ValueError):
            data = {}
    if not isinstance(data, dict):
        data = {}

    unsupported = data.get("unsupported_claims") or []
    omissions = data.get("omissions") or []
    if not isinstance(unsupported, list):
        unsupported = []
    if not isinstance(omissions, list):
        omissions = []

    faithful = data.get("faithful")
    if not isinstance(faithful, bool):
        faithful = None

    return {
        "faithful": faithful,
        "unsupported_claims": [str(item) for item in unsupported],
        "omissions": [str(item) for item in omissions],
    }


def is_unchanged(draft: str, refined: str) -> bool:
    """Whether the Refiner returned the draft unchanged (ignoring only outer whitespace)."""
    return draft.strip() == refined.strip()


def minimal_edit_ratio(draft: str, refined: str) -> float:
    """
    Similarity of the refined text to the draft in 0..1 (1.0 = identical).

    The Refiner is supposed to make targeted edits, so a well-behaved Refiner
    keeps this high. A low value means it rewrote more than it was asked to.
    """
    return difflib.SequenceMatcher(None, draft, refined).ratio()


def unsupported_removed_rate(
    draft: str,
    refined: str,
    unsupported_claims: list[str],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> float | None:
    """
    Fraction of flagged unsupported claims that no longer appear in the refined
    text. Returns None when nothing was flagged (no denominator to score).
    """
    if not unsupported_claims:
        return None
    removed = sum(
        1 for claim in unsupported_claims if not contains_phrase(refined, claim, threshold)
    )
    return removed / len(unsupported_claims)


def omissions_restored_rate(
    refined: str,
    omissions: list[str],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> float | None:
    """
    Fraction of flagged omissions that now appear in the refined text. Returns
    None when nothing was flagged (no denominator to score).
    """
    if not omissions:
        return None
    restored = sum(1 for fact in omissions if contains_phrase(refined, fact, threshold))
    return restored / len(omissions)


def evaluate_single(draft: str, refined: str, verdict: Any) -> dict:
    """
    Score one Refiner output against the draft it was given and the Verifier
    feedback that drove it. Returns a dict of the Refiner's contract metrics.

    `contract_pass` is the headline: it is True when the Refiner did the right
    thing for this case — left a faithful draft untouched, or removed every
    flagged claim and restored every flagged omission.
    """
    parsed = parse_verdict(verdict)
    unsupported = parsed["unsupported_claims"]
    omissions = parsed["omissions"]

    # A no-op case: the Verifier declared the draft faithful, or raised no flags.
    is_noop_case = parsed["faithful"] is True or (not unsupported and not omissions)

    removed_rate = unsupported_removed_rate(draft, refined, unsupported)
    restored_rate = omissions_restored_rate(refined, omissions)
    changed = not is_unchanged(draft, refined)

    if is_noop_case:
        noop_respected: bool | None = not changed
        contract_pass = not changed
    else:
        noop_respected = None
        removed_ok = removed_rate is None or removed_rate == 1.0
        restored_ok = restored_rate is None or restored_rate == 1.0
        contract_pass = removed_ok and restored_ok

    return {
        "changed": changed,
        "minimal_edit_ratio": minimal_edit_ratio(draft, refined),
        "num_unsupported_flagged": len(unsupported),
        "num_omissions_flagged": len(omissions),
        "unsupported_removed_rate": removed_rate,
        "omissions_restored_rate": restored_rate,
        "is_noop_case": is_noop_case,
        "noop_respected": noop_respected,
        "contract_pass": contract_pass,
    }


def evaluate_multiple(examples: list[dict]) -> list[dict]:
    """
    Score many Refiner outputs. Each example is a dict with keys `draft`,
    `refined`, and `verdict` (an optional `example_id` is carried through).
    Returns a list of per-example metric dicts.
    """
    results = []
    for example in examples:
        res = evaluate_single(example["draft"], example["refined"], example.get("verdict"))
        results.append({
            "example_id": example.get("example_id"),
            **res,
        })
    return results


def _mean(values: list[float]) -> float | None:
    """Mean of the values, or None when there is nothing to average."""
    return (sum(values) / len(values)) if values else None


def summarize_results(results: list[dict]) -> dict:
    """
    Summarize a list of per-example results into means and pass rates. Rates over
    a metric that was None (not applicable) for every example return None, never a
    fake zero, matching the rest of the FaithfulMed evaluation code.
    """
    if not results:
        return {
            "num_results": 0,
            "mean_minimal_edit_ratio": None,
            "mean_unsupported_removed_rate": None,
            "mean_omissions_restored_rate": None,
            "percent_contract_pass": None,
            "percent_noop_respected": None,
        }

    n = len(results)
    removed = [r["unsupported_removed_rate"] for r in results if r["unsupported_removed_rate"] is not None]
    restored = [r["omissions_restored_rate"] for r in results if r["omissions_restored_rate"] is not None]
    noop = [r["noop_respected"] for r in results if r["noop_respected"] is not None]

    return {
        "num_results": n,
        "mean_minimal_edit_ratio": _mean([r["minimal_edit_ratio"] for r in results]),
        "mean_unsupported_removed_rate": _mean(removed),
        "mean_omissions_restored_rate": _mean(restored),
        "percent_contract_pass": (sum(r["contract_pass"] for r in results) / n) * 100,
        "percent_noop_respected": (_mean([float(v) for v in noop]) * 100) if noop else None,
    }


def compare_texts(draft: str, refined: str) -> dict:
    """
    A before/after edit summary for one Refiner output: whether it changed the
    draft, how similar the two are, and the sentence-level additions and removals.
    """
    draft_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", draft.strip()) if s.strip()]
    refined_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", refined.strip()) if s.strip()]
    draft_set = set(draft_sentences)
    refined_set = set(refined_sentences)

    return {
        "before": draft,
        "after": refined,
        "changed": not is_unchanged(draft, refined),
        "minimal_edit_ratio": minimal_edit_ratio(draft, refined),
        "char_delta": len(refined) - len(draft),
        "word_delta": len(_tokens(refined)) - len(_tokens(draft)),
        "added_sentences": [s for s in refined_sentences if s not in draft_set],
        "removed_sentences": [s for s in draft_sentences if s not in refined_set],
    }


def contract_pass_rate(results: list[dict]) -> float:
    """
    Percentage of scored examples that honored the full Refiner contract.
    Returns -1.0 for an empty list, matching readability_eval's rate helpers.
    """
    if not results:
        return -1.0
    return (sum(r["contract_pass"] for r in results) / len(results)) * 100


def find_failures(results: list[dict]) -> list[dict]:
    """
    The examples that broke the Refiner contract, each with a short reason:
      - "changed a faithful draft" (should have been a no-op)
      - "left unsupported claims" (did not remove everything flagged)
      - "did not restore omissions" (did not add back everything flagged)
    """
    failures = []
    for res in results:
        if res["contract_pass"]:
            continue

        reasons = []
        if res["is_noop_case"] and res["changed"]:
            reasons.append("changed a faithful draft")
        if res["unsupported_removed_rate"] is not None and res["unsupported_removed_rate"] < 1.0:
            reasons.append("left unsupported claims")
        if res["omissions_restored_rate"] is not None and res["omissions_restored_rate"] < 1.0:
            reasons.append("did not restore omissions")

        failures.append({
            "example_id": res.get("example_id"),
            "reasons": reasons,
            "unsupported_removed_rate": res["unsupported_removed_rate"],
            "omissions_restored_rate": res["omissions_restored_rate"],
            "minimal_edit_ratio": res["minimal_edit_ratio"],
        })
    return failures
