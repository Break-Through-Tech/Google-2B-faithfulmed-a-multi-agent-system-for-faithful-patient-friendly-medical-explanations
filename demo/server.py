"""FastAPI server for the FaithfulMed demo.

Streams the real production pipeline to the browser as Server-Sent Events, or
replays a saved run from ``eval_results/system_demo_runs.jsonl`` with
compressed timing so the hosted demo works without spending Gemini quota.

Run locally (from the repository root):

    python -m uvicorn demo.server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import math
import os
from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents import LITELLM_PROVIDER_PREFIXES
from agents.production_pipeline import MAX_REVISION_CYCLES
from agents.verifier import VERIFIER_DEFAULT_MODEL
from evals.system_demo import AppendOnlyRunCache, SystemExample, load_system_examples

from .events import pipeline_done_event, public_stage, result_event, sse

ROOT = Path(__file__).resolve().parent.parent
DIST = Path(__file__).resolve().parent / "web" / "dist"
MODELS_PATH = ROOT / "evals" / "leaderboard_models.json"
LIVE_SPLITS = ("scratch", "dev")
MAX_SOURCE_CHARS = 15_000
KEEPALIVE_SECONDS = 15.0
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def load_env() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().removeprefix("export ").partition("=")
        if separator and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env()
MODEL = os.getenv("DEMO_MODEL") or VERIFIER_DEFAULT_MODEL
_live_slots = asyncio.Semaphore(max(1, int(os.getenv("DEMO_MAX_LIVE_RUNS", "1"))))

app = FastAPI(title="FaithfulMed demo")


def live_available() -> bool:
    """Live runs need Gemini credentials: the fact-check always runs on MODEL."""
    if os.getenv("DEMO_LIVE", "1") == "0":
        return False
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_USE_VERTEXAI"))


# ── Pipeline models ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def model_catalog() -> dict[str, dict[str, Any]]:
    """Models the pipeline can run on, from the leaderboard's model list."""
    models = json.loads(MODELS_PATH.read_text(encoding="utf-8"))["models"]
    return {
        key: {"key": key, "label": m["display_name"], "model_id": m["adk_model"], "provider": m["provider"],
              "openness": (m.get("openness") or {}).get("category"), "credential_env": m["credential_env"]}
        for key, m in models.items()
    }


def model_unavailable_reason(entry: dict[str, Any]) -> str | None:
    """None when this server can run the model, otherwise why it can't."""
    env = entry["credential_env"]
    if not (os.getenv(env) or (env == "GOOGLE_API_KEY" and os.getenv("GOOGLE_GENAI_USE_VERTEXAI"))):
        return f"Needs {env} on the server."
    if entry["model_id"].startswith(LITELLM_PROVIDER_PREFIXES) and importlib.util.find_spec("litellm") is None:
        return "Needs the litellm package on the server."
    return None


def default_model_key() -> str:
    catalog = model_catalog()
    return next((key for key, entry in catalog.items() if entry["model_id"] == MODEL), next(iter(catalog)))


# ── Saved runs ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def saved_runs() -> dict[str, dict[str, Any]]:
    """Latest completed run per example, keyed by run_id."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for record in AppendOnlyRunCache().records():
        if record.get("final_explanation") and record.get("pipeline_trace", {}).get("stages"):
            latest[(record.get("dataset", ""), record.get("example_id", ""))] = record
    return {record["run_id"]: record for record in latest.values()}


def _run_summary(record: dict[str, Any]) -> dict[str, Any]:
    stages = record["pipeline_trace"]["stages"]
    try:
        facts = len(json.loads(record["pipeline_trace"]["results"].get("extractor") or "[]"))
    except (json.JSONDecodeError, TypeError):
        facts = 0
    readability = record.get("readability_metrics") or {}
    factual = record.get("factual_metrics") or {}
    return {
        "run_id": record["run_id"],
        "dataset": record.get("dataset"),
        "split": record.get("split"),
        "example_id": record.get("example_id"),
        "created_at_utc": record.get("created_at_utc"),
        "model": record.get("pipeline_model"),
        "revisions": sum(stage.get("agent") == "Refiner" for stage in stages),
        "facts": facts,
        "fk_grade": readability.get("flesch_kincaid_grade"),
        "fidelity": factual.get("factual_fidelity_estimate"),
        "hallucination": factual.get("clinically_meaningful_hallucination"),
        "preview": " ".join((record.get("source_text") or "").split())[:180],
    }


@lru_cache(maxsize=1)
def live_examples() -> dict[str, SystemExample]:
    examples: dict[str, SystemExample] = {}
    for split in LIVE_SPLITS:
        for example in load_system_examples(split, ["MTSamples", "PLABA"], ROOT):
            examples[f"{example.dataset}/{split}/{example.example_id}"] = example
    return examples


# ── API ─────────────────────────────────────────────────────────────────

@app.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "live_available": live_available(),
        "model": MODEL,
        "evaluator_model": MODEL,
        "default_model": default_model_key(),
        "models": [
            {**{k: entry[k] for k in ("key", "label", "model_id", "provider", "openness")},
             "available": reason is None, "reason": reason}
            for entry in model_catalog().values()
            for reason in [model_unavailable_reason(entry)]
        ],
        "max_revision_cycles": MAX_REVISION_CYCLES,
        "max_source_chars": MAX_SOURCE_CHARS,
    }


@app.get("/api/runs")
def runs() -> list[dict[str, Any]]:
    summaries = [_run_summary(record) for record in saved_runs().values()]
    return sorted(summaries, key=lambda s: (-s["revisions"], s["facts"] == 0, s["dataset"] or "", s["example_id"] or ""))


@app.get("/api/examples")
def examples() -> list[dict[str, Any]]:
    return [
        {"key": key, "dataset": ex.dataset, "split": ex.split, "example_id": ex.example_id,
         "chars": len(ex.source_text), "source_text": ex.source_text}
        for key, ex in live_examples().items()
    ]


async def _stream_text(event: dict[str, Any], text: str, speed: float) -> AsyncIterator[str]:
    """Replay one recorded output in chunks; long outputs play faster, not longer."""
    duration = min(max(len(text) / 3500, 1.2), 5.0) / speed
    ticks = max(1, int(duration / 0.04))
    size = math.ceil(len(text) / ticks)
    for start in range(0, len(text), size):
        yield sse({**event, "delta": text[start:start + size]})
        await asyncio.sleep(0.04)


async def replay_events(record: dict[str, Any], speed: float) -> AsyncIterator[str]:
    trace = record["pipeline_trace"]
    yield sse({"type": "run_started", "meta": {
        "mode": "replay", "run_id": record["run_id"], "dataset": record.get("dataset"),
        "split": record.get("split"), "example_id": record.get("example_id"),
        "recorded_at_utc": record.get("created_at_utc"), "model": record.get("pipeline_model"),
        "source_text": record.get("source_text"),
    }})
    for stage in trace.get("stages", []):
        yield sse({"type": "stage_started", "stage": public_stage(stage, with_output=False)})
        output = stage.get("output") or ""
        if output and stage.get("agent") != "TermLookup":
            async for chunk in _stream_text({"type": "stage_chunk", "stage_id": stage["stage_id"]}, output, speed):
                yield chunk
        yield sse({"type": "stage_completed", "stage": public_stage(stage)})
        await asyncio.sleep(0.35 / speed)
    yield sse(pipeline_done_event(trace, record.get("final_explanation"), bool(record.get("accepted"))))
    raw = (record.get("factual_evaluator") or {}).get("raw_output")
    if raw:
        yield sse({"type": "evaluator_started"})
        async for chunk in _stream_text({"type": "evaluator_chunk"}, raw, speed):
            yield chunk
    yield sse(result_event(record))
    yield sse({"type": "done"})


@app.get("/api/replay/{run_id}")
async def replay(run_id: str, speed: float = Query(1.0, ge=0.25, le=8.0)) -> StreamingResponse:
    record = saved_runs().get(run_id)
    if record is None:
        raise HTTPException(404, "No saved run with that ID.")
    return StreamingResponse(replay_events(record, speed), media_type="text/event-stream", headers=SSE_HEADERS)


class RunRequest(BaseModel):
    source_text: str | None = Field(default=None, max_length=MAX_SOURCE_CHARS)
    example_key: str | None = None
    model: str | None = None


async def live_events(example: SystemExample, model: dict[str, Any]) -> AsyncIterator[str]:
    from evals.system_demo import run_system_example

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def on_pipeline_event(event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "stage_started":
            queue.put_nowait({"type": "stage_started", "stage": public_stage(event["record"], with_output=False)})
        elif kind == "stage_chunk":
            queue.put_nowait({"type": "stage_chunk", "stage_id": event["stage_id"], "delta": event["delta"]})
        elif kind in {"stage_completed", "stage_failed"}:
            queue.put_nowait({"type": "stage_completed", "stage": public_stage(event["record"])})
        elif kind in {"pipeline_completed", "pipeline_not_accepted", "pipeline_failed"}:
            trace = event["trace"]
            final = trace["results"].get("final_explanation")
            queue.put_nowait(pipeline_done_event(trace, final, trace.get("status") == "completed"))

    if _live_slots.locked():
        yield sse({"type": "error", "message": "A live run is already in progress."})
        yield sse({"type": "done"})
        return
    await _live_slots.acquire()
    task = asyncio.create_task(run_system_example(
        example, model["model_id"], MODEL, cache=None, persist=False,
        on_pipeline_event=on_pipeline_event,
        on_evaluator_start=lambda: queue.put_nowait({"type": "evaluator_started"}),
        on_evaluator_text=lambda delta, _acc: queue.put_nowait({"type": "evaluator_chunk", "delta": delta}),
    ))
    try:
        yield sse({"type": "run_started", "meta": {
            "mode": "live", "dataset": example.dataset, "split": example.split,
            "example_id": example.example_id, "model": model["model_id"], "model_label": model["label"],
            "evaluator_model": MODEL, "source_text": example.source_text,
        }})
        while True:
            getter = asyncio.ensure_future(queue.get())
            done, _ = await asyncio.wait({getter, task}, timeout=KEEPALIVE_SECONDS,
                                         return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                yield sse(getter.result())
                continue
            getter.cancel()
            if task in done:
                break
            yield ": keepalive\n\n"
        while not queue.empty():
            yield sse(queue.get_nowait())
        try:
            record = task.result()
        except Exception as exc:  # noqa: BLE001 - any pipeline failure must reach the browser
            yield sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        else:
            yield sse(result_event(record))
        yield sse({"type": "done"})
    finally:
        if not task.done():
            task.cancel()  # browser disconnected: stop spending quota
        _live_slots.release()


@app.post("/api/run")
async def run(request: RunRequest) -> StreamingResponse:
    if not live_available():
        raise HTTPException(503, "Live runs are disabled on this server. Use a saved run instead.")
    model = model_catalog().get(request.model or default_model_key())
    if model is None:
        raise HTTPException(422, "Unknown model.")
    if reason := model_unavailable_reason(model):
        raise HTTPException(503, f"{model['label']} isn't available on this server. {reason}")
    if request.example_key:
        example = live_examples().get(request.example_key)
        if example is None:
            raise HTTPException(404, "Unknown example.")
    else:
        text = (request.source_text or "").strip()
        if not text:
            raise HTTPException(422, "Paste a medical report or choose an example.")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        example = SystemExample("Pasted", "session", f"pasted-{digest}", text, {}, "session")
    if _live_slots.locked():
        raise HTTPException(429, "A live run is already in progress. Try a saved run, or wait a minute.")
    return StreamingResponse(live_events(example, model), media_type="text/event-stream", headers=SSE_HEADERS)


# ── Frontend ────────────────────────────────────────────────────────────

if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        target = (DIST / path).resolve()
        if path and target.is_file() and DIST in target.parents:
            return FileResponse(target)
        return FileResponse(DIST / "index.html")
