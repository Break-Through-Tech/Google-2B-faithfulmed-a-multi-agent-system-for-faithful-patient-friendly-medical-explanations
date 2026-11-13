"""FaithfulMed model leaderboard: result schema, manifests, aggregation, and reports.

Everything in this module is offline: it reads saved results and never calls a
model. Model runs live in ``evals.leaderboard_runner``.

Files (under eval_results/leaderboard/):
  manifests/<id>.json      shared list of notes every configuration must cover
  attempts.jsonl           one line per generation attempt (append-only)
  evaluations.jsonl        one line per automated-judge attempt (append-only)
  reviews/<sample>/        blind human-review items and the judgments CSV
  review_keys/<sample>.json  the unblinding key (keep away from reviewers)
  reports/<manifest>/      leaderboard.md, leaderboard.csv, notes.csv

Commands:
  python3 -m evals.leaderboard manifest --id dev_pilot --split dev --purpose pilot
  python3 -m evals.leaderboard report --manifest eval_results/leaderboard/manifests/dev_pilot.json
  python3 -m evals.leaderboard report --historical
  python3 -m evals.leaderboard review-sample --manifest <path> --per-config 5 --seed 20261110

Selection rules (documented in every report):
  * A note's result for a configuration and setup is its FIRST terminal attempt
    (completed or not accepted). Later terminal attempts are counted as
    duplicates and ignored, so reruns cannot be used to pick a better result.
  * Failed attempts stay in the file and in the report. A note with only failed
    attempts is reported as failed; a note with no attempts is reported missing.
  * Missing measurements are reported as unavailable, never as zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agents.baseline import BASELINE_MESSAGE, load_prompt as load_baseline_prompt
from agents.factual_evaluator import FACTUAL_EVALUATOR_PROMPT
from agents.production_pipeline import MAX_REVISION_CYCLES
from evals.system_demo import (
    METRIC_IMPLEMENTATION_VERSION,
    PIPELINE_IMPLEMENTATION_VERSION,
    aggregate_metrics,
    load_system_examples,
    prompt_hashes,
    select_examples,
)

ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD_DIR = ROOT / "eval_results" / "leaderboard"
MANIFEST_DIR = LEADERBOARD_DIR / "manifests"
ATTEMPTS_PATH = LEADERBOARD_DIR / "attempts.jsonl"
EVALUATIONS_PATH = LEADERBOARD_DIR / "evaluations.jsonl"
REVIEWS_DIR = LEADERBOARD_DIR / "reviews"
REVIEW_KEYS_DIR = LEADERBOARD_DIR / "review_keys"
REPORTS_DIR = LEADERBOARD_DIR / "reports"
MODELS_PATH = Path(__file__).resolve().parent / "leaderboard_models.json"
HISTORICAL_SYSTEM_PATH = ROOT / "eval_results" / "system_demo_runs.jsonl"
HISTORICAL_BASELINE_PATH = ROOT / "runs" / "baseline.jsonl"

MANIFEST_SCHEMA = "faithfulmed_leaderboard_manifest_v1"
ATTEMPT_SCHEMA = "faithfulmed_leaderboard_attempt_v1"
EVALUATION_SCHEMA = "faithfulmed_leaderboard_evaluation_v1"
REVIEW_ITEMS_SCHEMA = "faithfulmed_blind_review_items_v1"
REVIEW_KEY_SCHEMA = "faithfulmed_blind_review_key_v1"

COMPLETED, NOT_ACCEPTED, FAILED = "completed", "not_accepted", "failed"
TERMINAL = (COMPLETED, NOT_ACCEPTED)
JUDGE_VALID = ("scored", "no_claims", "insufficient_source")
PURPOSES = ("smoke", "pilot", "final_test", "other")
UNAVAILABLE = "unavailable"
PACKAGES = ("google-adk", "google-genai", "litellm")


# ── Small helpers ───────────────────────────────────────────────────────

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False))


def note_key(item: Dict[str, Any]) -> Tuple[str, str]:
    return (item["dataset"], item["example_id"])


def load_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    """Return (records, unreadable_line_count). A missing file is empty."""
    records: List[Dict[str, Any]] = []
    bad = 0
    if not Path(path).exists():
        return records, bad
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(value, dict):
            records.append(value)
        else:
            bad += 1
    return records, bad


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def percentile(values: List[float], fraction: float) -> Optional[float]:
    """Nearest-rank percentile; None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _pct(part: int, whole: int) -> Optional[float]:
    return part / whole * 100 if whole else None


# ── Catalog (models, prices, openness, configurations) ─────────────────

def load_catalog(path: Path = MODELS_PATH) -> Dict[str, Any]:
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    for config in catalog["configurations"]:
        if config["model_key"] not in catalog["models"]:
            raise ValueError(f"{config['config_id']} names an unknown model {config['model_key']!r}.")
        if config["type"] not in ("baseline", "full_system"):
            raise ValueError(f"{config['config_id']} has an invalid type {config['type']!r}.")
    return catalog


def configuration(catalog: Dict[str, Any], config_id: str) -> Dict[str, Any]:
    for config in catalog["configurations"]:
        if config["config_id"] == config_id:
            return {**config, "model": catalog["models"][config["model_key"]]}
    raise KeyError(f"Unknown configuration {config_id!r}.")


def configurations(catalog: Dict[str, Any], ids: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    wanted = None if ids is None else list(ids)
    if wanted:
        return [configuration(catalog, c) for c in wanted]
    return [configuration(catalog, c["config_id"]) for c in catalog["configurations"]]


def price_for(model: Dict[str, Any], on_date: str) -> Tuple[Optional[Dict[str, float]], str]:
    """Paid-tier list price in force on ``on_date`` (YYYY-MM-DD).

    Status is "ok", "no_paid_price" (the provider offers no paid tier, which is
    different from an unknown price), or "price_unknown".
    """
    pricing = model.get("pricing") or {}
    if pricing.get("paid_tier_available") is False:
        return None, "no_paid_price"
    for period in pricing.get("periods") or []:
        if (period.get("from") is None or on_date >= period["from"]) and \
                (period.get("to") is None or on_date <= period["to"]):
            return {"input": float(period["input"]), "output": float(period["output"])}, "ok"
    return None, "price_unknown"


def estimate_cost(tokens: Dict[str, Any], price: Optional[Dict[str, float]]) -> Optional[float]:
    if price is None:
        return None
    return tokens["input"] * price["input"] / 1e6 + tokens["output"] * price["output"] / 1e6


# ── Setups and fingerprints ────────────────────────────────────────────

def package_versions() -> Dict[str, Optional[str]]:
    import importlib.metadata as metadata

    versions: Dict[str, Optional[str]] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def generation_setup(config: Dict[str, Any], catalog: Dict[str, Any],
                     packages: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Any]:
    """Everything that can change a configuration's generated output.

    ``shared_setup_sha256`` covers the parts that must match across models of the
    same configuration type; ``setup_fingerprint`` adds the model itself.
    """
    common: Dict[str, Any] = {
        "config_type": config["type"],
        "generation_settings_policy": catalog.get("generation_settings_policy"),
        "packages": packages if packages is not None else package_versions(),
    }
    if config["type"] == "baseline":
        common.update(baseline_prompt_sha256=sha256_text(load_baseline_prompt()),
                      baseline_message=BASELINE_MESSAGE)
    else:
        hashes = {k: v for k, v in prompt_hashes().items() if k != "factual_evaluator"}
        common.update(pipeline_version=PIPELINE_IMPLEMENTATION_VERSION, prompt_hashes=hashes,
                      max_revision_cycles=MAX_REVISION_CYCLES, lookup=catalog["full_system"]["lookup"])
    model = config["model"]
    setup = {
        **common,
        "shared_setup_sha256": stable_hash(common),
        "model": {"model_key": config["model_key"], "model_id": model["model_id"],
                  "adk_model": model["adk_model"], "provider": model["provider"],
                  "adaptations": model.get("adaptations") or []},
    }
    setup["setup_fingerprint"] = stable_hash({"shared": setup["shared_setup_sha256"], "model": setup["model"],
                                              "config_id": config["config_id"]})
    return setup


def evaluator_setup(catalog: Dict[str, Any]) -> Dict[str, Any]:
    evaluator = catalog["evaluator"]
    setup = {
        "model_id": evaluator["model_id"],
        "judge_family": evaluator["judge_family"],
        "prompt_sha256": sha256_text(FACTUAL_EVALUATOR_PROMPT),
        "metric_version": METRIC_IMPLEMENTATION_VERSION,
    }
    setup["evaluator_fingerprint"] = stable_hash(setup)
    return setup


# ── Manifests ───────────────────────────────────────────────────────────

def manifest_note(example: Any) -> Dict[str, Any]:
    return {"dataset": example.dataset, "split": example.split, "example_id": example.example_id,
            "source_file": example.source_file, "source_sha256": sha256_text(example.source_text)}


def build_manifest(manifest_id: str, split: str, datasets: List[str], purpose: str,
                   limit: Optional[int] = None, root: Path = ROOT) -> Dict[str, Any]:
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}")
    if purpose == "final_test" and split != "test":
        raise ValueError("A final_test manifest must use the test split.")
    if split == "test" and purpose != "final_test":
        raise ValueError("The frozen test split is reserved for the final_test manifest.")
    examples = select_examples(load_system_examples(split, tuple(datasets), root), limit)
    notes = [manifest_note(e) for e in examples]
    return {"schema": MANIFEST_SCHEMA, "manifest_id": manifest_id, "purpose": purpose, "split": split,
            "datasets": list(datasets), "created_at_utc": now_utc(), "notes": notes,
            "manifest_sha256": stable_hash(notes)}


def load_manifest(path: Path) -> Dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{path} is not a leaderboard manifest.")
    if stable_hash(manifest["notes"]) != manifest.get("manifest_sha256"):
        raise ValueError(f"{path} was edited after creation (manifest_sha256 mismatch).")
    return manifest


def manifest_examples(manifest: Dict[str, Any], root: Path = ROOT) -> List[Any]:
    """Load the manifest's notes and verify each source text is unchanged."""
    by_split: Dict[str, Dict[Tuple[str, str], Any]] = {}
    out, problems = [], []
    for note in manifest["notes"]:
        split = note["split"]
        if split not in by_split:
            datasets = tuple(sorted({n["dataset"] for n in manifest["notes"] if n["split"] == split}))
            by_split[split] = {(e.dataset, e.example_id): e for e in load_system_examples(split, datasets, root)}
        example = by_split[split].get(note_key(note))
        if example is None:
            problems.append(f"{note['dataset']}/{note['example_id']}: not found")
        elif sha256_text(example.source_text) != note["source_sha256"]:
            problems.append(f"{note['dataset']}/{note['example_id']}: source text changed")
        else:
            out.append(example)
    if problems:
        raise ValueError("Manifest notes do not match local data: " + "; ".join(problems))
    return out


# ── Record builders ────────────────────────────────────────────────────

def summarize_calls(calls: List[Dict[str, Any]], *, latency_s: Optional[float]) -> Dict[str, Any]:
    """Token, retry, and timing totals for one attempt's model calls.

    Usage is complete only if every successful call reported token usage. Tokens
    from failed call attempts (rarely reported) are included in the totals.
    """
    ok = [c for c in calls if c.get("status") == "ok"]
    tokens = {
        "input": sum(int(c.get("input_tokens") or 0) for c in calls),
        "output": sum(int(c.get("output_tokens") or 0) for c in calls),
        "thinking": sum(int(c.get("thinking_tokens") or 0) for c in calls),
    }
    return {
        "latency_s": latency_s,
        "calls": calls,
        "model_calls_ok": len(ok),
        "model_calls_failed": len(calls) - len(ok),
        "retries": sum(1 for c in calls if c.get("status") != "ok" and c.get("retried")),
        "retry_wait_s": sum(float(c.get("wait_after_s") or 0.0) for c in calls),
        "tokens": tokens,
        "usage_complete": bool(ok) and all(c.get("usage_available") for c in ok),
    }


def new_attempt(config: Dict[str, Any], setup: Dict[str, Any], manifest: Dict[str, Any],
                note: Dict[str, Any], *, status: str, error: Optional[str], final_output: Optional[str],
                generation: Dict[str, Any], readability_metrics: Optional[Dict[str, Any]] = None,
                readability_error: Optional[str] = None, latest_draft: Optional[str] = None,
                pipeline_trace: Optional[Dict[str, Any]] = None, started_at_utc: Optional[str] = None,
                completed_at_utc: Optional[str] = None, attempt_id: Optional[str] = None) -> Dict[str, Any]:
    import uuid

    if status not in (COMPLETED, NOT_ACCEPTED, FAILED):
        raise ValueError(f"invalid attempt status {status!r}")
    model = config["model"]
    return {
        "schema": ATTEMPT_SCHEMA,
        "attempt_id": attempt_id or uuid.uuid4().hex,
        "source": "leaderboard",
        "config_id": config["config_id"],
        "config_type": config["type"],
        "model_key": config["model_key"],
        "model_id": model["model_id"],
        "adk_model": model["adk_model"],
        "provider": model["provider"],
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "note": {k: note[k] for k in ("dataset", "split", "example_id", "source_file", "source_sha256")},
        "setup": setup,
        "setup_fingerprint": setup["setup_fingerprint"],
        "status": status,
        "error": error,
        "final_output": final_output,
        "latest_draft": latest_draft,
        "readability_metrics": readability_metrics,
        "readability_error": readability_error,
        "generation": generation,
        "pipeline_trace": pipeline_trace,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc or now_utc(),
    }


def new_evaluation(attempt: Dict[str, Any], evaluator: Dict[str, Any], *, status: str,
                   raw_output: Optional[str], parsed_output: Optional[Dict[str, Any]],
                   factual_metrics: Optional[Dict[str, Any]], usage: Dict[str, Any],
                   error: Optional[str]) -> Dict[str, Any]:
    import uuid

    return {
        "schema": EVALUATION_SCHEMA,
        "evaluation_id": uuid.uuid4().hex,
        "attempt_id": attempt["attempt_id"],
        "config_id": attempt["config_id"],
        "note": attempt["note"],
        "evaluator": evaluator,
        "evaluator_fingerprint": evaluator["evaluator_fingerprint"],
        "status": status,
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "factual_metrics": factual_metrics,
        "usage": usage,
        "error": error,
        "created_at_utc": now_utc(),
    }


# ── Selection and aggregation ──────────────────────────────────────────

def select_notes(attempts: List[Dict[str, Any]], manifest_notes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply the selection rules for one configuration-and-setup group."""
    expected = {note_key(n): n["source_sha256"] for n in manifest_notes}
    by_note: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    stale = 0
    for attempt in attempts:
        key = note_key(attempt["note"])
        if key not in expected:
            continue
        if attempt["note"].get("source_sha256") != expected[key]:
            stale += 1
            continue
        by_note[key].append(attempt)
    notes: Dict[Tuple[str, str], Dict[str, Any]] = {}
    duplicates = failed_attempts = 0
    for note in manifest_notes:
        key = note_key(note)
        tries = sorted(by_note.get(key, []), key=lambda a: (a.get("started_at_utc") or "", a.get("completed_at_utc") or ""))
        terminal = [a for a in tries if a.get("status") in TERMINAL]
        failed = [a for a in tries if a.get("status") == FAILED]
        duplicates += max(len(terminal) - 1, 0)
        failed_attempts += len(failed)
        chosen = terminal[0] if terminal else None
        state = chosen["status"] if chosen else (FAILED if failed else "missing")
        notes[key] = {"state": state, "chosen": chosen, "failed": failed,
                      "last_error": (failed[-1].get("error") if failed else None)}
    return {"notes": notes, "duplicates": duplicates, "failed_attempts": failed_attempts, "stale_attempts": stale}


def choose_evaluation(evaluations: List[Dict[str, Any]], fingerprint: str) -> Optional[Dict[str, Any]]:
    """First valid judge response for this attempt and evaluator; else the latest failure."""
    mine = sorted((e for e in evaluations if e.get("evaluator_fingerprint") == fingerprint),
                  key=lambda e: e.get("created_at_utc") or "")
    valid = [e for e in mine if e.get("status") in JUDGE_VALID]
    if valid:
        return valid[0]
    return mine[-1] if mine else None


def summarize_group(config: Dict[str, Any], setup_fingerprint: Optional[str], attempts: List[Dict[str, Any]],
                    evaluations_by_attempt: Dict[str, List[Dict[str, Any]]], manifest: Dict[str, Any],
                    evaluator_fp: str, catalog: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """One leaderboard row plus per-note status rows."""
    model = config["model"]
    selection = select_notes(attempts, manifest["notes"])
    notes = selection["notes"]
    counts = Counter(n["state"] for n in notes.values())
    chosen = [n["chosen"] for n in notes.values() if n["chosen"]]
    completed = [a for a in chosen if a["status"] == COMPLETED]

    # Faithfulness (automated judge) and readability.
    judged: List[Dict[str, Any]] = []
    evaluation_states: Counter = Counter()
    evaluator_tokens = {"input": 0, "output": 0, "thinking": 0}
    evaluator_calls = 0
    for attempt in completed:
        evaluation = choose_evaluation(evaluations_by_attempt.get(attempt["attempt_id"], []), evaluator_fp)
        if evaluation is None:
            evaluation_states["not_evaluated"] += 1
        else:
            evaluation_states[evaluation["status"]] += 1
        for item in evaluations_by_attempt.get(attempt["attempt_id"], []):
            if item.get("evaluator_fingerprint") == evaluator_fp:
                usage = item.get("usage") or {}
                for k in evaluator_tokens:
                    evaluator_tokens[k] += int((usage.get("tokens") or {}).get(k) or 0)
                evaluator_calls += int(usage.get("model_calls_ok") or 0) + int(usage.get("model_calls_failed") or 0)
        judged.append({"readability_metrics": attempt.get("readability_metrics"),
                       "factual_metrics": (evaluation or {}).get("factual_metrics")
                       if evaluation and evaluation.get("status") in JUDGE_VALID else None})
    agg = aggregate_metrics(judged) if judged else aggregate_metrics([])
    scored = evaluation_states["scored"]

    # Latency, retries, tokens, cost (generation only; the judge is reported separately).
    gen = [a.get("generation") or {} for a in chosen]
    latencies = [float(g["latency_s"]) for g in gen if g.get("latency_s") is not None]
    full_usage = [a for a in chosen if (a.get("generation") or {}).get("usage_complete")]
    tokens_in = [a["generation"]["tokens"]["input"] for a in full_usage]
    tokens_out = [a["generation"]["tokens"]["output"] for a in full_usage]
    tokens_think = [a["generation"]["tokens"]["thinking"] for a in full_usage]
    costs, cost_states = [], Counter()
    for attempt in chosen:
        g = attempt.get("generation") or {}
        price, state = price_for(model, (attempt.get("started_at_utc") or now_utc())[:10])
        if not g.get("usage_complete"):
            cost_states["usage_incomplete"] += 1
        elif state != "ok":
            cost_states[state] += 1
        else:
            costs.append(estimate_cost(g["tokens"], price))
            cost_states["estimated"] += 1
    failed_tokens = {"input": 0, "output": 0}
    for n in notes.values():
        for attempt in n["failed"]:
            t = (attempt.get("generation") or {}).get("tokens") or {}
            failed_tokens["input"] += int(t.get("input") or 0)
            failed_tokens["output"] += int(t.get("output") or 0)
    if not chosen:
        cost_status = UNAVAILABLE
    elif cost_states["estimated"] == len(chosen):
        cost_status = "estimated"
    elif cost_states["no_paid_price"] == len(chosen):
        cost_status = "no_paid_price"
    else:
        cost_status = "partial: " + ", ".join(f"{k}={v}" for k, v in sorted(cost_states.items()))
    evaluator_model = catalog["models"][catalog["evaluator"]["model_key"]]
    eval_price, eval_price_state = price_for(evaluator_model, now_utc()[:10])
    openness = model.get("openness") or {}

    row = {
        "config_id": config["config_id"],
        "display_name": model["display_name"],
        "model_id": model["model_id"],
        "provider": model["provider"],
        "config_type": config["type"],
        "setup": (setup_fingerprint or "")[:12] or "none",
        "setup_fingerprint": setup_fingerprint,
        "notes_in_manifest": len(manifest["notes"]),
        "completed": counts[COMPLETED],
        "not_accepted": counts[NOT_ACCEPTED],
        "failed": counts[FAILED],
        "missing": counts["missing"],
        "completion_pct": _pct(counts[COMPLETED], len(manifest["notes"])),
        "duplicate_attempts_ignored": selection["duplicates"],
        "failed_attempts_total": selection["failed_attempts"],
        "stale_attempts_ignored": selection["stale_attempts"],
        "judge_scored": scored,
        "judge_no_claims_or_insufficient": evaluation_states["no_claims"] + evaluation_states["insufficient_source"],
        "judge_failed": sum(v for k, v in evaluation_states.items() if k not in JUDGE_VALID and k != "not_evaluated"),
        "judge_not_run": evaluation_states["not_evaluated"],
        "scoring_coverage_pct": _pct(scored, len(completed)),
        "fidelity_pct": agg["factual_fidelity_estimate"],
        "claims_assessed": agg["claims_assessed"],
        "hallucination_rate_pct": agg["hallucination_rate"],
        "hallucination_outputs_scored": agg["hallucination_outputs_scored"],
        "readability_scored": agg["readability_scored"],
        "fk_grade_mean": agg["mean_flesch_kincaid_grade"],
        "pct_at_or_below_grade_8": agg["percent_at_or_below_grade_8"],
        "smog_mean": agg["mean_smog"],
        "jargon_density_mean": agg["mean_jargon_density"],
        "word_count_mean": agg["mean_word_count"],
        "refusal_rate_pct": agg["refusal_rate"],
        "latency_n": len(latencies),
        "latency_median_s": statistics.median(latencies) if latencies else None,
        "latency_p90_s": percentile(latencies, 0.9),
        "latency_mean_s": _mean(latencies),
        "retries_total": sum(int(g.get("retries") or 0) for g in gen),
        "retry_wait_total_s": sum(float(g.get("retry_wait_s") or 0.0) for g in gen),
        "usage_coverage": f"{len(full_usage)}/{len(chosen)}",
        "gen_input_tokens_mean": _mean(tokens_in),
        "gen_output_tokens_mean": _mean(tokens_out),
        "gen_thinking_tokens_mean": _mean(tokens_think),
        "gen_input_tokens_total": sum(tokens_in) if full_usage else None,
        "gen_output_tokens_total": sum(tokens_out) if full_usage else None,
        "failed_attempt_input_tokens": failed_tokens["input"],
        "failed_attempt_output_tokens": failed_tokens["output"],
        "est_cost_per_note_usd": _mean(costs),
        "est_cost_total_usd": sum(costs) if costs else None,
        "cost_status": cost_status,
        "billed": f"$0 ({catalog.get('billing_tier', 'unknown')} tier)" if catalog.get("billing_tier") == "free" else UNAVAILABLE,
        "judge_input_tokens": evaluator_tokens["input"] if evaluator_calls else None,
        "judge_output_tokens": evaluator_tokens["output"] if evaluator_calls else None,
        "judge_model_calls": evaluator_calls,
        "judge_est_cost_usd": estimate_cost(evaluator_tokens, eval_price) if evaluator_calls and eval_price_state == "ok" else None,
        "openness_category": openness.get("category"),
        "weights": openness.get("weights"),
        "license": openness.get("license"),
        "hosting": openness.get("hosting"),
        "openness_sources": " ".join(openness.get("sources") or []),
        "adaptations": " | ".join(model.get("adaptations") or []) or "none",
    }
    note_rows = []
    for note in manifest["notes"]:
        n = notes[note_key(note)]
        chosen_attempt = n["chosen"]
        evaluation = choose_evaluation(evaluations_by_attempt.get(chosen_attempt["attempt_id"], []), evaluator_fp) \
            if chosen_attempt else None
        note_rows.append({
            "config_id": config["config_id"], "setup": row["setup"], "dataset": note["dataset"],
            "example_id": note["example_id"], "state": n["state"],
            "attempt_id": chosen_attempt["attempt_id"] if chosen_attempt else None,
            "failed_attempts": len(n["failed"]),
            "judge_status": (evaluation or {}).get("status") if chosen_attempt and chosen_attempt["status"] == COMPLETED else None,
            "error": (chosen_attempt or {}).get("error") or n["last_error"],
        })
    return row, note_rows


def build_report(manifest: Dict[str, Any], catalog: Dict[str, Any], attempts: List[Dict[str, Any]],
                 evaluations: List[Dict[str, Any]], *, review_keys: Optional[List[Dict[str, Any]]] = None,
                 judgments: Optional[Dict[str, List[Dict[str, str]]]] = None) -> Dict[str, Any]:
    evaluator = evaluator_setup(catalog)
    evaluations_by_attempt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    other_evaluator = 0
    for e in evaluations:
        evaluations_by_attempt[e.get("attempt_id")].append(e)
        if e.get("evaluator_fingerprint") != evaluator["evaluator_fingerprint"]:
            other_evaluator += 1
    leaderboard_attempts = [a for a in attempts if a.get("schema") == ATTEMPT_SCHEMA]
    rows, note_rows, warnings = [], [], []
    for config in configurations(catalog):
        mine = [a for a in leaderboard_attempts if a.get("config_id") == config["config_id"]]
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for a in mine:
            groups[a["setup_fingerprint"]].append(a)
        # Groups with no attempts on this manifest's notes are not shown.
        wanted = {note_key(n) for n in manifest["notes"]}
        groups = {fp: items for fp, items in groups.items() if any(note_key(a["note"]) in wanted for a in items)}
        if not groups:
            row, notes = summarize_group(config, None, [], evaluations_by_attempt, manifest,
                                         evaluator["evaluator_fingerprint"], catalog)
            rows.append({**row, "_shared": None})
            note_rows.extend(notes)
            continue
        if len(groups) > 1:
            warnings.append(f"{config['config_id']} has results from {len(groups)} different setups; each is a "
                            "separate row and they are not comparable with each other.")
        for fp, items in sorted(groups.items(), key=lambda kv: min(a.get("started_at_utc") or "" for a in kv[1])):
            row, notes = summarize_group(config, fp, items, evaluations_by_attempt, manifest,
                                         evaluator["evaluator_fingerprint"], catalog)
            rows.append({**row, "_shared": items[0]["setup"].get("shared_setup_sha256")})
            note_rows.extend(notes)
    for kind in ("baseline", "full_system"):
        shared = {r["_shared"] for r in rows if r["config_type"] == kind and r["_shared"]}
        if len(shared) > 1:
            warnings.append(f"{kind} rows were generated with different prompts, pipeline, lookup, or package "
                            "versions (shared setup differs); compare them with caution.")
    if other_evaluator:
        warnings.append(f"{other_evaluator} evaluation records used a different judge model or prompt and were ignored.")
    for r in rows:
        r.pop("_shared", None)
        if r["missing"] or r["failed"]:
            warnings.append(f"{r['config_id']} ({r['setup']}): {r['missing']} missing and {r['failed']} failed "
                            f"of {r['notes_in_manifest']} manifest notes.")
    report = {
        "generated_at_utc": now_utc(),
        "manifest": {k: manifest.get(k) for k in ("manifest_id", "purpose", "split", "datasets", "manifest_sha256")},
        "note_count": len(manifest["notes"]),
        "evaluator": {**evaluator, "note": catalog["evaluator"]["note"]},
        "billing_note": catalog.get("billing_note"),
        "generation_settings_note": catalog.get("generation_settings_note"),
        "lookup_note": catalog["full_system"].get("lookup_note"),
        "rows": rows,
        "notes": note_rows,
        "warnings": warnings,
        "human_review": [],
    }
    if review_keys:
        attempt_index = {a["attempt_id"]: a for a in leaderboard_attempts}
        for key in review_keys:
            report["human_review"].append(human_review_summary(
                key, (judgments or {}).get(key["sample_id"], []), evaluations_by_attempt,
                evaluator["evaluator_fingerprint"], attempt_index))
    return report


# ── Blind human review ─────────────────────────────────────────────────

REVIEW_INSTRUCTIONS = (
    "For each item, read the source note and the patient-facing explanation. Decide whether the "
    "explanation contains a clinically meaningful claim that the source does not support or that "
    "contradicts it. Answer yes, no, or unsure in judgments.csv. Model and configuration names are "
    "hidden on purpose; do not open the review key."
)


def draw_review_sample(report: Dict[str, Any], attempts: List[Dict[str, Any]], manifest: Dict[str, Any],
                       *, per_config: int, seed: int, sample_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Equal-sized random sample of completed outputs from every configuration."""
    if per_config < 1:
        raise ValueError("per_config must be at least 1.")
    by_id = {a["attempt_id"]: a for a in attempts}
    per_config_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in report["notes"]:
        per_config_rows[row["config_id"]].append(row)
    setups = Counter((r["config_id"]) for r in report["rows"] if r["setup"] != "none")
    multi = [c for c, n in setups.items() if n > 1]
    if multi:
        raise ValueError(f"Configurations with more than one setup cannot be sampled: {multi}.")
    rng = random.Random(seed)
    chosen: List[Tuple[str, Dict[str, Any]]] = []
    shortfall = []
    for row in report["rows"]:
        config_id = row["config_id"]
        eligible = sorted((r for r in per_config_rows[config_id] if r["state"] == COMPLETED),
                          key=lambda r: (r["dataset"], r["example_id"]))
        if len(eligible) < per_config:
            shortfall.append(f"{config_id} has {len(eligible)} completed outputs")
            continue
        chosen.extend((config_id, by_id[r["attempt_id"]]) for r in rng.sample(eligible, per_config))
    if shortfall:
        raise ValueError(f"Cannot draw {per_config} per configuration: " + "; ".join(shortfall))
    rng.shuffle(chosen)
    items, key_items = [], {}
    sources = {(e.dataset, e.example_id): e.source_text for e in manifest_examples(manifest)}
    for index, (config_id, attempt) in enumerate(chosen, start=1):
        review_id = f"R-{index:03d}"
        items.append({"review_id": review_id, "dataset": attempt["note"]["dataset"],
                      "example_id": attempt["note"]["example_id"],
                      "source_text": sources[note_key(attempt["note"])],
                      "explanation": attempt["final_output"]})
        key_items[review_id] = {"config_id": config_id, "attempt_id": attempt["attempt_id"],
                                "setup_fingerprint": attempt["setup_fingerprint"],
                                "dataset": attempt["note"]["dataset"], "example_id": attempt["note"]["example_id"]}
    created = now_utc()
    review_items = {"schema": REVIEW_ITEMS_SCHEMA, "sample_id": sample_id, "created_at_utc": created,
                    "instructions": REVIEW_INSTRUCTIONS, "items": items}
    key = {"schema": REVIEW_KEY_SCHEMA, "sample_id": sample_id, "created_at_utc": created, "seed": seed,
           "per_config": per_config, "manifest_id": manifest["manifest_id"],
           "manifest_sha256": manifest["manifest_sha256"],
           "evaluator_fingerprint": report["evaluator"]["evaluator_fingerprint"],
           "configurations": [r["config_id"] for r in report["rows"]], "items": key_items}
    return review_items, key


def review_markdown(review_items: Dict[str, Any]) -> str:
    lines = [f"# Blind review sample {review_items['sample_id']}", "", review_items["instructions"], ""]
    for item in review_items["items"]:
        lines += [f"## {item['review_id']}", "", "**Source note**", "", item["source_text"], "",
                  "**Explanation**", "", item["explanation"] or "", ""]
    return "\n".join(lines)


def judgments_template(review_items: Dict[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["review_id", "reviewer", "clinically_meaningful_hallucination", "notes"])
    for item in review_items["items"]:
        writer.writerow([item["review_id"], "", "", ""])
    return buffer.getvalue()


def _human_label(value: str) -> Optional[str]:
    value = (value or "").strip().lower()
    if value in ("yes", "y", "true"):
        return "yes"
    if value in ("no", "n", "false"):
        return "no"
    if value in ("unsure", "u", "?"):
        return "unsure"
    return None


def load_judgments(path: Path) -> List[Dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle)]


def cohen_kappa_binary(pairs: List[Tuple[str, str]]) -> Optional[float]:
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    p_yes_a = sum(a == "yes" for a, _ in pairs) / n
    p_yes_b = sum(b == "yes" for _, b in pairs) / n
    expected = p_yes_a * p_yes_b + (1 - p_yes_a) * (1 - p_yes_b)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def human_review_summary(key: Dict[str, Any], judgment_rows: List[Dict[str, str]],
                         evaluations_by_attempt: Dict[str, List[Dict[str, Any]]], evaluator_fp: str,
                         attempt_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Human judgments and automated-judge labels, side by side, never merged."""
    per_config: Dict[str, Dict[str, Any]] = {c: {"sampled": 0, "human": [], "unsure": 0, "unlabelled": 0,
                                                "judge": [], "pairs": []} for c in key["configurations"]}
    labels_by_item: Dict[str, List[str]] = defaultdict(list)
    for row in judgment_rows:
        label = _human_label(row.get("clinically_meaningful_hallucination", ""))
        if row.get("review_id") in key["items"] and label:
            labels_by_item[row["review_id"]].append(label)
    for review_id, item in key["items"].items():
        stats = per_config.setdefault(item["config_id"], {"sampled": 0, "human": [], "unsure": 0, "unlabelled": 0,
                                                          "judge": [], "pairs": []})
        stats["sampled"] += 1
        evaluation = choose_evaluation(evaluations_by_attempt.get(item["attempt_id"], []), evaluator_fp)
        judge_value = ((evaluation or {}).get("factual_metrics") or {}).get("clinically_meaningful_hallucination")
        judge = None if judge_value is None else ("yes" if judge_value else "no")
        if judge:
            stats["judge"].append(judge)
        labels = labels_by_item.get(review_id, [])
        if not labels:
            stats["unlabelled"] += 1
        for label in labels:
            if label == "unsure":
                stats["unsure"] += 1
                continue
            stats["human"].append(label)
            if judge:
                stats["pairs"].append((label, judge))
    rows = []
    for config_id, s in per_config.items():
        rows.append({
            "config_id": config_id, "sampled": s["sampled"], "human_judgments": len(s["human"]),
            "human_unsure": s["unsure"], "unlabelled_items": s["unlabelled"],
            "human_hallucination_rate_pct": _pct(sum(h == "yes" for h in s["human"]), len(s["human"])),
            "judge_hallucination_rate_pct": _pct(sum(j == "yes" for j in s["judge"]), len(s["judge"])),
            "paired": len(s["pairs"]),
            "agreement_pct": _pct(sum(a == b for a, b in s["pairs"]), len(s["pairs"])),
            "cohen_kappa": cohen_kappa_binary(s["pairs"]),
        })
    return {"sample_id": key["sample_id"], "seed": key["seed"], "per_config": key["per_config"], "rows": rows}


# ── Historical runs (labelled with their original settings) ────────────

def historical_rows(splits: Tuple[str, ...] = ("scratch", "dev"),
                    system_path: Path = HISTORICAL_SYSTEM_PATH,
                    baseline_path: Path = HISTORICAL_BASELINE_PATH) -> List[Dict[str, Any]]:
    """Summaries of saved pre-leaderboard runs. Not comparable with leaderboard rows."""
    rows = []
    system, _ = load_jsonl(system_path)
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in system:
        if r.get("split") not in splits:
            continue
        prompt_set = stable_hash(r.get("prompt_hashes") or {})[:8]
        groups[(r.get("pipeline_model"), r.get("pipeline_version"), prompt_set)].append(r)
    for (model, version, prompt_set), records in sorted(groups.items()):
        first: Dict[Tuple[str, str], Dict[str, Any]] = {}
        duplicates = 0
        for r in sorted(records, key=lambda x: x.get("created_at_utc") or ""):
            k = (r.get("dataset"), r.get("example_id"))
            if k in first:
                duplicates += 1
            else:
                first[k] = r
        chosen = list(first.values())
        agg = aggregate_metrics(chosen)
        latencies = []
        for r in chosen:
            t = r.get("pipeline_trace") or {}
            if t.get("started_at_utc") and t.get("completed_at_utc"):
                latencies.append((datetime.fromisoformat(t["completed_at_utc"]) -
                                  datetime.fromisoformat(t["started_at_utc"])).total_seconds())
        evaluators = sorted({str(r.get("evaluator_model")) for r in chosen})
        rows.append({
            "label": f"historical full system · {model} · {version} · prompt set {prompt_set}",
            "original_settings": f"system demo cache; judge {', '.join(evaluators)}; splits "
                                 f"{', '.join(sorted({r.get('split') for r in chosen}))}; no token usage recorded",
            "notes": len(chosen), "duplicates_ignored": duplicates,
            "with_output": sum(bool(r.get("final_explanation")) for r in chosen),
            "fidelity_pct": agg["factual_fidelity_estimate"],
            "hallucination_rate_pct": agg["hallucination_rate"],
            "fk_grade_mean": agg["mean_flesch_kincaid_grade"],
            "pct_at_or_below_grade_8": agg["percent_at_or_below_grade_8"],
            "latency_median_s": statistics.median(latencies) if latencies else None,
            "latency_basis": "trace wall time (demo streaming; no retry data)",
            "est_cost_per_note_usd": None,
        })
    baseline, _ = load_jsonl(baseline_path)
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in baseline:
        if r.get("split") in splits:
            by_model[str(r.get("model"))].append(r)
    for model, records in sorted(by_model.items()):
        ok = [r for r in records if r.get("status") == "success"]
        agg = aggregate_metrics([{"readability_metrics": r.get("metrics")} for r in ok])
        rows.append({
            "label": f"historical baseline · {model}",
            "original_settings": "runs/baseline.jsonl; prompt version, timing, and token usage not recorded; "
                                 f"splits {', '.join(sorted({r.get('split') for r in records}))}",
            "notes": len(records), "duplicates_ignored": 0, "with_output": len(ok),
            "fidelity_pct": None, "hallucination_rate_pct": None,
            "fk_grade_mean": agg["mean_flesch_kincaid_grade"],
            "pct_at_or_below_grade_8": agg["percent_at_or_below_grade_8"],
            "latency_median_s": None, "latency_basis": "not recorded", "est_cost_per_note_usd": None,
        })
    return rows


# ── Rendering ───────────────────────────────────────────────────────────

def fmt(value: Any, digits: int = 1, unit: str = "") -> str:
    if value is None:
        return UNAVAILABLE
    if isinstance(value, float):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def fmt_usd(value: Optional[float]) -> str:
    return UNAVAILABLE if value is None else f"${value:.4f}"


def _table(headers: List[str], rows: List[List[str]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in r) + " |" for r in rows]
    return out


def render_markdown(report: Dict[str, Any], historical: Optional[List[Dict[str, Any]]] = None) -> str:
    m, ev = report["manifest"], report["evaluator"]
    lines = [f"# FaithfulMed leaderboard: {m['manifest_id']}", "",
             f"Manifest `{m['manifest_id']}` ({m['purpose']}, split {m['split']}, {report['note_count']} notes, "
             f"sha256 {m['manifest_sha256'][:12]}). Generated {report['generated_at_utc']}.", "",
             f"**Automated faithfulness judge:** {ev['model_id']} ({ev['judge_family']}), factual-evaluator prompt "
             f"{ev['prompt_sha256'][:12]}, metrics {ev['metric_version']}. {ev['note']}", ""]
    for note in (report.get("generation_settings_note"), report.get("lookup_note"), report.get("billing_note")):
        if note:
            lines += [f"- {note}"]
    lines.append("")
    if report["warnings"]:
        lines += ["## Warnings", ""] + [f"- {w}" for w in report["warnings"]] + [""]
    rows = report["rows"]
    lines += ["## Coverage", ""] + _table(
        ["Configuration", "Setup", "Notes", "Completed", "Not accepted", "Failed", "Missing", "Judge scored",
         "Scoring coverage", "Failed attempts", "Duplicates ignored"],
        [[f"{r['config_id']} ({r['display_name']})", r["setup"], r["notes_in_manifest"], r["completed"],
          r["not_accepted"], r["failed"], r["missing"], r["judge_scored"], fmt(r["scoring_coverage_pct"], 0, "%"),
          r["failed_attempts_total"], r["duplicate_attempts_ignored"]] for r in rows]) + [""]
    lines += ["## Faithfulness and readability", "",
              "Faithfulness comes from the automated Gemini judge on completed outputs only. Not-accepted and "
              "failed notes have no output and are not scored.", ""] + _table(
        ["Configuration", "Fidelity (supported claims)", "Hallucination rate", "Outputs judged", "FK grade",
         "At or below grade 8", "SMOG", "Jargon density", "Words", "Refusals"],
        [[r["config_id"], fmt(r["fidelity_pct"], 1, "%"), fmt(r["hallucination_rate_pct"], 1, "%"),
          r["hallucination_outputs_scored"], fmt(r["fk_grade_mean"]), fmt(r["pct_at_or_below_grade_8"], 0, "%"),
          fmt(r["smog_mean"]), fmt(r["jargon_density_mean"], 2), fmt(r["word_count_mean"], 0),
          fmt(r["refusal_rate_pct"], 0, "%")] for r in rows]) + [""]
    lines += ["## Latency, tokens, and cost (generation only)", "",
              "Latency is wall time for generation, including retries and their waits, excluding the judge. "
              "Tokens are per completed or not-accepted note. Estimated cost uses the paid list price on the run date.",
              ""] + _table(
        ["Configuration", "Median latency", "p90 latency", "Retries (wait)", "Usage coverage", "Input tokens/note",
         "Output tokens/note (thinking)", "Est. cost/note", "Est. total", "Cost status", "Billed",
         "Tokens in failed attempts (in/out)"],
        [[r["config_id"], fmt(r["latency_median_s"], 1, " s"), fmt(r["latency_p90_s"], 1, " s"),
          f"{r['retries_total']} ({r['retry_wait_total_s']:.1f} s)", r["usage_coverage"],
          fmt(r["gen_input_tokens_mean"], 0), f"{fmt(r['gen_output_tokens_mean'], 0)} ({fmt(r['gen_thinking_tokens_mean'], 0)})",
          fmt_usd(r["est_cost_per_note_usd"]), fmt_usd(r["est_cost_total_usd"]), r["cost_status"], r["billed"],
          f"{r['failed_attempt_input_tokens']}/{r['failed_attempt_output_tokens']}"] for r in rows]) + [""]
    lines += ["## Automated judge usage (reported separately)", ""] + _table(
        ["Configuration", "Judge calls", "Judge input tokens", "Judge output tokens", "Judge est. cost"],
        [[r["config_id"], r["judge_model_calls"], fmt(r["judge_input_tokens"]), fmt(r["judge_output_tokens"]),
          fmt_usd(r["judge_est_cost_usd"])] for r in rows]) + [""]
    seen = set()
    open_rows = []
    for r in rows:
        if r["model_id"] in seen:
            continue
        seen.add(r["model_id"])
        open_rows.append([f"{r['display_name']} (`{r['model_id']}`)", r["openness_category"], r["weights"],
                          r["license"], r["hosting"], r["openness_sources"]])
    lines += ["## Openness", ""] + _table(["Model", "Category", "Weights", "License", "Hosting", "Sources"],
                                          open_rows) + [""]
    adapt = [[r["config_id"], r["adaptations"]] for r in rows if r["config_type"] == "full_system"]
    lines += ["## Model-specific adaptations", ""] + _table(["Configuration", "Adaptations"], adapt) + [""]
    problems = [n for n in report["notes"] if n["state"] in (FAILED, "missing", NOT_ACCEPTED)]
    lines += ["## Notes not completed (against the manifest)", ""]
    lines += (_table(["Configuration", "Setup", "Note", "State", "Failed attempts", "Last error"],
                     [[n["config_id"], n["setup"], f"{n['dataset']}/{n['example_id']}", n["state"],
                       n["failed_attempts"], (n["error"] or "")[:160]] for n in problems])
              if problems else ["Every configuration completed every manifest note."]) + [""]
    for review in report.get("human_review") or []:
        lines += [f"## Blind human review: {review['sample_id']}", "",
                  f"Seed {review['seed']}, {review['per_config']} outputs per configuration. Human judgments and "
                  "the automated judge are shown side by side and are never merged.", ""] + _table(
            ["Configuration", "Sampled", "Human judgments", "Unsure", "Unlabelled", "Human hallucination rate",
             "Judge hallucination rate (same items)", "Paired", "Agreement", "Cohen's kappa"],
            [[r["config_id"], r["sampled"], r["human_judgments"], r["human_unsure"], r["unlabelled_items"],
              fmt(r["human_hallucination_rate_pct"], 0, "%"), fmt(r["judge_hallucination_rate_pct"], 0, "%"),
              r["paired"], fmt(r["agreement_pct"], 0, "%"), fmt(r["cohen_kappa"], 2)] for r in review["rows"]]) + [""]
    if historical is not None:
        lines += ["## Historical runs (original settings; not comparable with the rows above)", "",
                  "These came from earlier work with different pipeline versions, prompts, and logging. "
                  "They show the report format only.", ""] + _table(
            ["Run group", "Original settings", "Notes", "With output", "Duplicates ignored", "Fidelity",
             "Hallucination rate", "FK grade", "At or below grade 8", "Median latency", "Latency basis", "Est. cost"],
            [[h["label"], h["original_settings"], h["notes"], h["with_output"], h["duplicates_ignored"],
              fmt(h["fidelity_pct"], 1, "%"), fmt(h["hallucination_rate_pct"], 1, "%"), fmt(h["fk_grade_mean"]),
              fmt(h["pct_at_or_below_grade_8"], 0, "%"), fmt(h["latency_median_s"], 0, " s"), h["latency_basis"],
              fmt_usd(h["est_cost_per_note_usd"])] for h in historical]) + [""]
    lines += ["## Method", "",
              "- One result per configuration, setup, and note: the first completed or not-accepted attempt. "
              "Later duplicates are counted and ignored; failed attempts are kept and counted.",
              "- Fidelity = supported claims / assessed claims, pooled over judged outputs. Hallucination rate = "
              "outputs with at least one clinically meaningful unsupported or contradicted claim / outputs judged.",
              "- Readability uses textstat on the final output (Flesch-Kincaid grade, SMOG, Dale-Chall difficult-word share).",
              "- Output tokens include thinking/reasoning tokens (computed as total minus input). Usage coverage "
              "counts notes whose every successful model call reported usage; cost is unavailable otherwise.",
              "- 'no_paid_price' means the provider offers no paid tier for the model (cost is not zero and not unknown).",
              "- 'unavailable' means the measurement does not exist; it is never treated as zero.", ""]
    return "\n".join(lines)


ROW_FIELDS = [
    "config_id", "display_name", "model_id", "provider", "config_type", "setup", "setup_fingerprint",
    "notes_in_manifest", "completed", "not_accepted", "failed", "missing", "completion_pct",
    "duplicate_attempts_ignored", "failed_attempts_total", "stale_attempts_ignored", "judge_scored",
    "judge_no_claims_or_insufficient", "judge_failed", "judge_not_run", "scoring_coverage_pct", "fidelity_pct",
    "claims_assessed", "hallucination_rate_pct", "hallucination_outputs_scored", "readability_scored",
    "fk_grade_mean", "pct_at_or_below_grade_8", "smog_mean", "jargon_density_mean", "word_count_mean",
    "refusal_rate_pct", "latency_n", "latency_median_s", "latency_p90_s", "latency_mean_s", "retries_total",
    "retry_wait_total_s", "usage_coverage", "gen_input_tokens_mean", "gen_output_tokens_mean",
    "gen_thinking_tokens_mean", "gen_input_tokens_total", "gen_output_tokens_total",
    "failed_attempt_input_tokens", "failed_attempt_output_tokens", "est_cost_per_note_usd", "est_cost_total_usd",
    "cost_status", "billed", "judge_model_calls", "judge_input_tokens", "judge_output_tokens",
    "judge_est_cost_usd", "openness_category", "weights", "license", "hosting", "openness_sources", "adaptations",
]
NOTE_FIELDS = ["config_id", "setup", "dataset", "example_id", "state", "attempt_id", "failed_attempts",
               "judge_status", "error"]


def to_csv(rows: List[Dict[str, Any]], fields: List[str]) -> str:
    """CSV text; unavailable values are empty cells (documented in the Markdown report)."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})
    return buffer.getvalue()


def write_report(report: Dict[str, Any], out_dir: Path, historical: Optional[List[Dict[str, Any]]] = None) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "leaderboard.md", out_dir / "leaderboard.csv", out_dir / "notes.csv"]
    paths[0].write_text(render_markdown(report, historical), encoding="utf-8")
    paths[1].write_text(to_csv(report["rows"], ROW_FIELDS), encoding="utf-8")
    paths[2].write_text(to_csv(report["notes"], NOTE_FIELDS), encoding="utf-8")
    if historical is not None:
        hist_fields = list(historical[0].keys()) if historical else ["label"]
        path = out_dir / "historical.csv"
        path.write_text(to_csv(historical, hist_fields), encoding="utf-8")
        paths.append(path)
    return paths


def historical_only_markdown(historical: List[Dict[str, Any]]) -> str:
    fake_report = {"manifest": {"manifest_id": "historical", "purpose": "other", "split": "-",
                                "manifest_sha256": "-" * 12},
                   "note_count": 0, "generated_at_utc": now_utc(),
                   "evaluator": {"model_id": "as recorded", "judge_family": "Gemini", "prompt_sha256": "as recorded",
                                 "metric_version": "as recorded",
                                 "note": "Historical rows keep their original judge and settings."},
                   "rows": [], "notes": [], "warnings": [], "human_review": []}
    text = render_markdown(fake_report, historical)
    # Drop the empty leaderboard sections; keep the header, historical table, and method.
    head, _, rest = text.partition("## Coverage")
    hist = rest[rest.index("## Historical runs"):]
    return head + hist


# ── CLI ─────────────────────────────────────────────────────────────────

def _review_keys_for(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    keys = []
    for path in sorted(REVIEW_KEYS_DIR.glob("*.json")):
        key = json.loads(path.read_text(encoding="utf-8"))
        if key.get("schema") == REVIEW_KEY_SCHEMA and key.get("manifest_sha256") == manifest["manifest_sha256"]:
            keys.append(key)
    return keys


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="FaithfulMed leaderboard (offline commands).")
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("manifest", help="Create a shared evaluation manifest from local data.")
    m.add_argument("--id", required=True)
    m.add_argument("--split", required=True, choices=("scratch", "dev", "test"))
    m.add_argument("--purpose", required=True, choices=PURPOSES)
    m.add_argument("--datasets", nargs="+", default=["MTSamples", "PLABA"])
    m.add_argument("--limit", type=int, default=None)
    m.add_argument("--out", type=Path, default=None)

    r = sub.add_parser("report", help="Write the Markdown and CSV leaderboard from saved results.")
    r.add_argument("--manifest", type=Path)
    r.add_argument("--historical", action="store_true", help="Add (or only show) labelled historical runs.")
    r.add_argument("--historical-splits", nargs="+", default=["scratch", "dev"])
    r.add_argument("--attempts", type=Path, default=ATTEMPTS_PATH)
    r.add_argument("--evaluations", type=Path, default=EVALUATIONS_PATH)
    r.add_argument("--models", type=Path, default=MODELS_PATH)
    r.add_argument("--out-dir", type=Path, default=None)

    s = sub.add_parser("review-sample", help="Draw a blind, equal-sized human-review sample.")
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--per-config", type=int, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--sample-id", default=None)
    s.add_argument("--attempts", type=Path, default=ATTEMPTS_PATH)
    s.add_argument("--evaluations", type=Path, default=EVALUATIONS_PATH)
    s.add_argument("--models", type=Path, default=MODELS_PATH)
    args = parser.parse_args(argv)

    if args.command == "manifest":
        manifest = build_manifest(args.id, args.split, args.datasets, args.purpose, args.limit)
        out = args.out or MANIFEST_DIR / f"{args.id}.json"
        if out.exists():
            print(f"{out} already exists; manifests are immutable. Choose a new --id.")
            return 1
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {out} ({len(manifest['notes'])} notes, sha256 {manifest['manifest_sha256'][:12]}).")
        return 0

    catalog = load_catalog(args.models)
    attempts, bad_attempts = load_jsonl(args.attempts)
    evaluations, bad_evaluations = load_jsonl(args.evaluations)

    if args.command == "report":
        historical = historical_rows(tuple(args.historical_splits)) if args.historical else None
        if args.manifest is None:
            if historical is None:
                parser.error("report needs --manifest, --historical, or both.")
            out_dir = args.out_dir or REPORTS_DIR / "historical"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "leaderboard.md").write_text(historical_only_markdown(historical), encoding="utf-8")
            (out_dir / "historical.csv").write_text(to_csv(historical, list(historical[0].keys())), encoding="utf-8")
            print(f"Wrote {out_dir / 'leaderboard.md'} and {out_dir / 'historical.csv'}.")
            return 0
        manifest = load_manifest(args.manifest)
        keys = _review_keys_for(manifest)
        judgments = {k["sample_id"]: load_judgments(REVIEWS_DIR / k["sample_id"] / "judgments.csv") for k in keys}
        report = build_report(manifest, catalog, attempts, evaluations, review_keys=keys, judgments=judgments)
        if bad_attempts or bad_evaluations:
            report["warnings"].append(f"Skipped unreadable lines: {bad_attempts} in attempts, "
                                      f"{bad_evaluations} in evaluations.")
        paths = write_report(report, args.out_dir or REPORTS_DIR / manifest["manifest_id"], historical)
        print("Wrote " + ", ".join(str(p) for p in paths))
        return 0

    manifest = load_manifest(args.manifest)
    report = build_report(manifest, catalog, attempts, evaluations)
    sample_id = args.sample_id or f"{manifest['manifest_id']}-seed{args.seed}"
    items_dir = REVIEWS_DIR / sample_id
    key_path = REVIEW_KEYS_DIR / f"{sample_id}.json"
    if items_dir.exists() or key_path.exists():
        print(f"Sample {sample_id} already exists; choose another --sample-id.")
        return 1
    review_items, key = draw_review_sample(report, attempts, manifest, per_config=args.per_config,
                                           seed=args.seed, sample_id=sample_id)
    items_dir.mkdir(parents=True)
    (items_dir / "items.json").write_text(json.dumps(review_items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (items_dir / "items.md").write_text(review_markdown(review_items), encoding="utf-8")
    (items_dir / "judgments.csv").write_text(judgments_template(review_items), encoding="utf-8")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(review_items['items'])} blind items to {items_dir} and the key to {key_path}. "
          "Give reviewers only items.md and judgments.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
