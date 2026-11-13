"""Resumable leaderboard runner. The ``run`` command CALLS MODELS AND USES API QUOTA.

  python3 -m evals.leaderboard_runner status --manifest eval_results/leaderboard/manifests/smoke_dev.json
  python3 -m evals.leaderboard_runner run    --manifest eval_results/leaderboard/manifests/smoke_dev.json
  python3 -m evals.leaderboard_runner run    --manifest .../dev_pilot.json --retry-failed
  python3 -m evals.leaderboard_runner run    --manifest .../test_final.json --final

For every manifest note and configuration it generates once (baseline or full
system, with the configuration's model powering every generation agent), then
asks the fixed Gemini judge to score the output. Each generation attempt is
appended to attempts.jsonl and each judge attempt to evaluations.jsonl as soon
as it finishes, so an interrupted run resumes where it stopped.

Resuming never repeats completed work. A note whose attempts all failed is
skipped unless --retry-failed is given, and is retried at most
--max-note-attempts times in total; every failed attempt stays in the file.
Each model call is retried at most --max-call-attempts times, only for rate
limits, server errors, and timeouts; retry waits count toward latency.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from agents import run_agent_once
from agents.baseline import run_baseline
from agents.factual_evaluator import run_factual_evaluator
from agents.medical_lookup import MedicalLookup, UMLSClient
from agents.production_pipeline import run_production_pipeline
from agents.verifier import PRODUCTION_VERIFIER_MESSAGE, create_verifier_agent, production_verifier_prompt
from evals.leaderboard import (
    ATTEMPTS_PATH, COMPLETED, EVALUATIONS_PATH, FAILED, JUDGE_VALID, MODELS_PATH, NOT_ACCEPTED, ROOT, TERMINAL,
    append_jsonl, choose_evaluation, configuration, configurations, evaluator_setup, generation_setup, load_catalog,
    load_jsonl, load_manifest, manifest_examples, new_attempt, new_evaluation, note_key, now_utc,
    package_versions, summarize_calls,
)
from evals.system_demo import factual_metrics, output_metrics, parse_factual_evaluation

RETRYABLE_CODES = {429, 500, 502, 503, 504}
RETRYABLE_NAMES = {"RateLimitError", "ServiceUnavailableError", "InternalServerError", "APIConnectionError",
                   "Timeout", "APITimeoutError", "TimeoutError", "ServerError"}
RETRYABLE_TEXT = re.compile(r"\b(429|500|502|503|504)\b|RESOURCE_EXHAUSTED|UNAVAILABLE|overloaded", re.I)


def is_retryable(exc: BaseException) -> bool:
    """Rate limits, server errors, and timeouts are retried; everything else is not.

    A numeric HTTP code, when present, decides on its own (a 400 is never retried).
    """
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value in RETRYABLE_CODES
    if type(exc).__name__ in RETRYABLE_NAMES or isinstance(exc, asyncio.TimeoutError):
        return True
    return bool(RETRYABLE_TEXT.search(str(exc)))


class CallRecorder:
    """Runs model calls with bounded retries and records usage and timing per attempt."""

    def __init__(self, *, max_attempts: int = 3, base_delay: float = 2.0, max_delay: float = 60.0,
                 clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 rng: Optional[random.Random] = None):
        self.max_attempts = max(1, max_attempts)
        self.base_delay, self.max_delay = base_delay, max_delay
        self.clock, self.sleep = clock, sleep
        self.rng = rng or random.Random()
        self.calls: List[Dict[str, Any]] = []

    async def call(self, stage: str, fn: Callable[[Callable[[Dict[str, Any]], None]], Awaitable[str]]) -> str:
        for attempt in range(1, self.max_attempts + 1):
            usages: List[Dict[str, Any]] = []
            started = self.clock()
            record: Dict[str, Any] = {"stage": stage, "attempt": attempt}
            try:
                text = await fn(usages.append)
            except Exception as exc:  # recorded, then retried or re-raised
                retryable = is_retryable(exc)
                will_retry = retryable and attempt < self.max_attempts
                record.update(self._usage(usages), status="error", error=f"{type(exc).__name__}: {exc}"[:500],
                              retryable=retryable, retried=will_retry, elapsed_s=self.clock() - started,
                              wait_after_s=0.0)
                self.calls.append(record)
                if not will_retry:
                    raise
                delay = min(self.max_delay, self.base_delay * 2 ** (attempt - 1)) + self.rng.uniform(0, 0.5)
                record["wait_after_s"] = delay
                await self.sleep(delay)
                continue
            record.update(self._usage(usages), status="ok", error=None, retryable=None, retried=False,
                          elapsed_s=self.clock() - started, wait_after_s=0.0)
            self.calls.append(record)
            return text
        raise RuntimeError("unreachable")  # pragma: no cover

    @staticmethod
    def _usage(usages: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "usage_available": bool(usages),
            "responses_with_usage": len(usages),
            "input_tokens": sum(u["input_tokens"] for u in usages),
            "output_tokens": sum(u["output_tokens"] for u in usages),
            "thinking_tokens": sum(u.get("thinking_tokens", 0) for u in usages),
            "usage_method": sorted({u.get("method", "") for u in usages}),
        }


# ── Generation and judging (real model calls) ──────────────────────────

GenerationResult = Dict[str, Any]


async def generate_baseline(config: Dict[str, Any], example: Any, recorder: CallRecorder,
                            lookup: Optional[MedicalLookup]) -> GenerationResult:
    model = config["model"]["adk_model"]
    text = await recorder.call("Baseline", lambda on_usage: run_baseline(example.source_text, model,
                                                                         on_usage=on_usage))
    return {"status": COMPLETED, "final_output": text, "error": None, "latest_draft": None, "trace": None}


async def generate_system(config: Dict[str, Any], example: Any, recorder: CallRecorder,
                          lookup: Optional[MedicalLookup]) -> GenerationResult:
    """The production pipeline with every generation agent on this configuration's model."""
    model = config["model"]["adk_model"]

    async def executor(agent: Any, state: Dict[str, Any], message: str, on_text: Callable[[str, str], None]) -> str:
        text = await recorder.call(agent.name, lambda on_usage: run_agent_once(
            agent, state=state, message=message, on_usage=on_usage))
        on_text(text, text)
        return text

    async def verifier_runner(source: str, facts: str, draft: str, model_id: str,
                              on_text: Callable[[str, str], None], definitions: str) -> str:
        prompt = production_verifier_prompt(source, facts, draft, definitions)
        text = await recorder.call("Verifier", lambda on_usage: run_agent_once(
            create_verifier_agent(model_id), state={"evaluation_prompt": prompt},
            message=PRODUCTION_VERIFIER_MESSAGE, on_usage=on_usage))
        on_text(text, text)
        return text

    result = await run_production_pipeline(example.source_text, model, agent_executor=executor,
                                           verifier_runner=verifier_runner, lookup=lookup)
    trace = result.trace
    if result.accepted:
        status = COMPLETED
    elif trace.get("status") == "not_accepted":
        status = NOT_ACCEPTED
    else:
        status = FAILED
    return {"status": status, "final_output": result.final_explanation, "error": trace.get("error"),
            "latest_draft": trace.get("latest_draft"), "trace": trace}


GENERATORS = {"baseline": generate_baseline, "full_system": generate_system}


async def judge_output(attempt: Dict[str, Any], example: Any, evaluator: Dict[str, Any],
                       recorder: CallRecorder) -> Tuple[str, Optional[str], Optional[Dict[str, Any]],
                                                        Optional[Dict[str, Any]], Optional[str]]:
    """The fixed Gemini factual evaluator. Returns (status, raw, parsed, metrics, error)."""
    final = attempt["final_output"]
    try:
        raw = await recorder.call("FactualEvaluator", lambda on_usage: run_factual_evaluator(
            example.source_text, final, example.references, evaluator["model_id"], on_usage=on_usage))
    except Exception as exc:
        return "failed", None, None, None, f"{type(exc).__name__}: {exc}"[:500]
    try:
        parsed = parse_factual_evaluation(raw, final, example.source_text)
    except Exception as exc:
        return "parse_error", raw, None, None, f"{type(exc).__name__}: {exc}"[:500]
    return parsed["status"], raw, parsed, factual_metrics(parsed), None


# ── Resume planning (offline) ──────────────────────────────────────────

def plan_work(manifest: Dict[str, Any], setups: Dict[str, Dict[str, Any]], attempts: List[Dict[str, Any]],
              evaluations: List[Dict[str, Any]], evaluator_fp: str, *, retry_failed: bool,
              max_note_attempts: int, evaluate: bool = True) -> List[Dict[str, Any]]:
    """Decide what each (note, configuration) needs, without calling any model.

    Actions: generate, evaluate, done, failed_skipped (failed; not retrying),
    failed_exhausted (attempt limit reached), judge_exhausted.
    """
    evaluations_by_attempt: Dict[str, List[Dict[str, Any]]] = {}
    for e in evaluations:
        evaluations_by_attempt.setdefault(e.get("attempt_id"), []).append(e)
    tasks = []
    for note in manifest["notes"]:
        for config_id, setup in setups.items():
            mine = sorted((a for a in attempts
                           if a.get("config_id") == config_id and a.get("setup_fingerprint") == setup["setup_fingerprint"]
                           and note_key(a["note"]) == note_key(note)
                           and a["note"].get("source_sha256") == note["source_sha256"]),
                          key=lambda a: a.get("started_at_utc") or "")
            terminal = [a for a in mine if a.get("status") in TERMINAL]
            failed = [a for a in mine if a.get("status") == FAILED]
            task = {"note": note, "config_id": config_id, "attempt": None, "prior_failures": len(failed)}
            if terminal:
                chosen = terminal[0]
                task["attempt"] = chosen
                if chosen["status"] != COMPLETED or not evaluate:
                    task["action"] = "done"
                else:
                    mine_evals = [e for e in evaluations_by_attempt.get(chosen["attempt_id"], [])
                                  if e.get("evaluator_fingerprint") == evaluator_fp]
                    valid = choose_evaluation(mine_evals, evaluator_fp)
                    if valid and valid.get("status") in JUDGE_VALID:
                        task["action"] = "done"
                    elif len(mine_evals) >= max_note_attempts:
                        task["action"] = "judge_exhausted"
                    else:
                        task["action"] = "evaluate"
            elif failed:
                if len(failed) >= max_note_attempts:
                    task["action"] = "failed_exhausted"
                else:
                    task["action"] = "generate" if retry_failed else "failed_skipped"
            else:
                task["action"] = "generate"
            tasks.append(task)
    return tasks


# ── Orchestration ──────────────────────────────────────────────────────

Generator = Callable[[Dict[str, Any], Any, CallRecorder, Optional[MedicalLookup]], Awaitable[GenerationResult]]
Judge = Callable[[Dict[str, Any], Any, Dict[str, Any], CallRecorder], Awaitable[Tuple[Any, ...]]]


async def run_leaderboard(manifest: Dict[str, Any], catalog: Dict[str, Any], config_ids: Optional[List[str]], *,
                          examples: List[Any], attempts_path: Path = ATTEMPTS_PATH,
                          evaluations_path: Path = EVALUATIONS_PATH, retry_failed: bool = False,
                          max_note_attempts: int = 3, evaluate: bool = True,
                          lookup: Optional[MedicalLookup] = None,
                          generators: Optional[Dict[str, Generator]] = None, judge: Judge = judge_output,
                          recorder_factory: Callable[[], CallRecorder] = CallRecorder,
                          packages: Optional[Dict[str, Optional[str]]] = None,
                          clock: Callable[[], float] = time.perf_counter,
                          log: Callable[[str], None] = print) -> Dict[str, int]:
    generators = generators or GENERATORS
    configs = {c["config_id"]: c for c in configurations(catalog, config_ids)}
    packages = packages if packages is not None else package_versions()
    setups = {cid: generation_setup(c, catalog, packages) for cid, c in configs.items()}
    evaluator = evaluator_setup(catalog)
    by_note = {(e.dataset, e.example_id): e for e in examples}
    attempts, _ = load_jsonl(attempts_path)
    evaluations, _ = load_jsonl(evaluations_path)
    tasks = plan_work(manifest, setups, attempts, evaluations, evaluator["evaluator_fingerprint"],
                      retry_failed=retry_failed, max_note_attempts=max_note_attempts, evaluate=evaluate)
    counts: Dict[str, int] = {}
    for task in tasks:
        counts[task["action"]] = counts.get(task["action"], 0) + 1
    log("Plan: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    done = {"generated": 0, "generation_failed": 0, "not_accepted": 0, "judged": 0, "judge_failed": 0}
    for index, task in enumerate(tasks, start=1):
        if task["action"] not in ("generate", "evaluate"):
            continue
        config = configs[task["config_id"]]
        note = task["note"]
        example = by_note[note_key(note)]
        label = f"[{index}/{len(tasks)}] {config['config_id']} · {note['dataset']}/{note['example_id']}"
        attempt = task["attempt"]
        if task["action"] == "generate":
            recorder = recorder_factory()
            started_at = now_utc()
            start = clock()
            try:
                result = await generators[config["type"]](config, example, recorder, lookup)
            except Exception as exc:
                result = {"status": FAILED, "final_output": None, "latest_draft": None, "trace": None,
                          "error": f"{type(exc).__name__}: {exc}"[:500]}
            latency = clock() - start
            readability, readability_error = (output_metrics(result["final_output"])
                                              if result.get("final_output") else (None, None))
            attempt = new_attempt(config, setups[config["config_id"]], manifest, note,
                                  status=result["status"], error=result.get("error"),
                                  final_output=result.get("final_output"), latest_draft=result.get("latest_draft"),
                                  generation=summarize_calls(recorder.calls, latency_s=latency),
                                  readability_metrics=readability, readability_error=readability_error,
                                  pipeline_trace=result.get("trace"), started_at_utc=started_at)
            append_jsonl(attempts_path, attempt)
            key = {COMPLETED: "generated", NOT_ACCEPTED: "not_accepted"}.get(attempt["status"], "generation_failed")
            done[key] += 1
            log(f"{label}: {attempt['status']} in {latency:.1f} s"
                + (f" ({attempt['error']})" if attempt["error"] else ""))
            if attempt["status"] != COMPLETED or not evaluate:
                continue
        recorder = recorder_factory()
        start = clock()
        status, raw, parsed, metrics, error = await judge(attempt, example, evaluator, recorder)
        usage = summarize_calls(recorder.calls, latency_s=clock() - start)
        append_jsonl(evaluations_path, new_evaluation(attempt, evaluator, status=status, raw_output=raw,
                                                      parsed_output=parsed, factual_metrics=metrics,
                                                      usage=usage, error=error))
        done["judged" if status in JUDGE_VALID else "judge_failed"] += 1
        log(f"{label}: judge {status}" + (f" ({error})" if error else ""))
    return done


# ── Preflight and CLI ──────────────────────────────────────────────────

def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines into the environment without overriding existing values."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        key = key.replace("export ", "").strip()
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key, value.strip().strip("'\""))


def pinned_versions(path: Path = ROOT / "requirements.txt") -> Dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*([A-Za-z0-9_.-]+)==([^\s#]+)", line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def preflight(catalog: Dict[str, Any], config_ids: List[str], env: Dict[str, str]) -> List[str]:
    """Problems that must be fixed before any model call."""
    problems = []
    needed = {configuration(catalog, c)["model"]["credential_env"] for c in config_ids}
    needed.add(catalog["models"][catalog["evaluator"]["model_key"]]["credential_env"])
    problems += [f"{name} is not set" for name in sorted(needed) if not env.get(name)]
    types = {configuration(catalog, c)["type"] for c in config_ids}
    if "full_system" in types and catalog["full_system"]["lookup"].get("umls") and not env.get("UMLS_API_KEY"):
        problems.append("UMLS_API_KEY is not set (full-system runs use UMLS lookups)")
    return problems


def build_lookup(catalog: Dict[str, Any], env: Dict[str, str]) -> MedicalLookup:
    """The lookup configuration declared in the catalog, the same for every model."""
    expected = catalog["full_system"]["lookup"]
    return MedicalLookup(UMLSClient(env["UMLS_API_KEY"]) if expected.get("umls") else None)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="FaithfulMed leaderboard runner.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--configs", nargs="+", default=["all"])
        p.add_argument("--retry-failed", action="store_true")
        p.add_argument("--max-note-attempts", type=int, default=3)
        p.add_argument("--no-evaluate", action="store_true", help="Generate only; judge later.")
        p.add_argument("--attempts", type=Path, default=ATTEMPTS_PATH)
        p.add_argument("--evaluations", type=Path, default=EVALUATIONS_PATH)
        p.add_argument("--models", type=Path, default=MODELS_PATH)
        if name == "run":
            p.add_argument("--final", action="store_true", help="Required for the frozen test manifest.")
            p.add_argument("--max-call-attempts", type=int, default=3)
    args = parser.parse_args(argv)

    load_dotenv()
    catalog = load_catalog(args.models)
    config_ids = [c["config_id"] for c in catalog["configurations"]] if args.configs == ["all"] else args.configs
    manifest = load_manifest(args.manifest)
    packages = package_versions()
    setups = {c: generation_setup(configuration(catalog, c), catalog, packages) for c in config_ids}
    evaluator = evaluator_setup(catalog)

    if args.command == "status":
        attempts, _ = load_jsonl(args.attempts)
        evaluations, _ = load_jsonl(args.evaluations)
        tasks = plan_work(manifest, setups, attempts, evaluations, evaluator["evaluator_fingerprint"],
                          retry_failed=args.retry_failed, max_note_attempts=args.max_note_attempts,
                          evaluate=not args.no_evaluate)
        table: Dict[str, Dict[str, int]] = {}
        for t in tasks:
            table.setdefault(t["config_id"], {}).setdefault(t["action"], 0)
            table[t["config_id"]][t["action"]] += 1
        for config_id in config_ids:
            print(f"{config_id:18s} setup {setups[config_id]['setup_fingerprint'][:12]}  "
                  + ", ".join(f"{k}={v}" for k, v in sorted(table.get(config_id, {}).items())))
        return 0

    if manifest["purpose"] == "final_test" and not args.final:
        print("This is the frozen test manifest. Re-run with --final once the setup is fixed.")
        return 2
    pins = pinned_versions()
    drift = {k: (v, pins.get(k)) for k, v in packages.items() if pins.get(k) and v != pins.get(k)}
    if drift:
        print("Warning: installed versions differ from requirements.txt pins "
              + ", ".join(f"{k} {have} (pinned {want})" for k, (have, want) in drift.items())
              + ". Results will form a separate setup.")
    problems = preflight(catalog, config_ids, dict(os.environ))
    if problems:
        print("BLOCKED: " + "; ".join(problems))
        return 2
    examples = manifest_examples(manifest)
    types = {configuration(catalog, c)["type"] for c in config_ids}
    lookup = build_lookup(catalog, dict(os.environ)) if "full_system" in types else None
    print(f"Manifest {manifest['manifest_id']} ({len(manifest['notes'])} notes) × {len(config_ids)} configurations; "
          f"judge {evaluator['model_id']}. This uses API quota.")
    result = asyncio.run(run_leaderboard(
        manifest, catalog, config_ids, examples=examples, attempts_path=args.attempts,
        evaluations_path=args.evaluations, retry_failed=args.retry_failed,
        max_note_attempts=args.max_note_attempts, evaluate=not args.no_evaluate, lookup=lookup,
        recorder_factory=lambda: CallRecorder(max_attempts=args.max_call_attempts), packages=packages))
    print("Done: " + ", ".join(f"{k}={v}" for k, v in result.items()))
    print(f"Report: python3 -m evals.leaderboard report --manifest {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
