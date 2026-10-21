"""Run and save Baseline readability evaluations."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
import random
from google.genai import errors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.baseline import (
    BASELINE_DEFAULT_MODEL,
    run_baseline,
)
from evals.readability_eval import evaluate_single

ANNOTATIONS_DIR = ROOT / "data" / "annotations"
RESULTS_FILE = ROOT / "runs" / "baseline.jsonl"

VALID_SPLITS = ("dev", "scratch", "test")


def load_samples(split: str) -> list[dict[str, Any]]:
    """Load source documents from one adjudicated dataset split."""

    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown dataset split: {split}")

    folder = ANNOTATIONS_DIR / split / "adjudicated"
    samples = []

    for path in sorted(folder.glob("*.json")):
        with path.open(encoding="utf-8") as file:
            sample = json.load(file)

        source_text = sample.get("source_text")

        if not isinstance(source_text, str) or not source_text.strip():
            continue

        samples.append(
            {
                "example_id": sample.get("example_id", path.stem),
                "source_text": source_text,
                "source_file": str(path.relative_to(ROOT)),
            }
        )

    return samples


def load_completed_ids(split: str, model: str) -> set[str]:
    """Find samples that already have successful results."""

    completed = set()

    if not RESULTS_FILE.exists():
        return completed

    with RESULTS_FILE.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if (
                record.get("split") == split
                and record.get("model") == model
                and record.get("status") == "success"
            ):
                completed.add(str(record["example_id"]))

    return completed


def save_record(record: dict[str, Any]) -> None:
    """Append one evaluation record immediately."""

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


async def run_baseline_with_retry(
    source_text: str,
    *,
    model: str,
    max_attempts: int = 4,
    initial_delay: float = 1.0,
) -> str:
    """Retry temporary Gemini server errors with exponential backoff."""

    retryable_codes = {500, 502, 503, 504}

    for attempt in range(max_attempts):
        try:
            return await run_baseline(source_text, model=model)

        except errors.ServerError as exc:
            if (
                getattr(exc, "code", None) not in retryable_codes
                or attempt == max_attempts - 1
            ):
                raise

            delay = initial_delay * (2 ** attempt)
            delay += random.uniform(0, 0.5)

            print(
                f"Gemini server error {exc.code}. "
                f"Retrying attempt {attempt + 2}/{max_attempts} "
                f"in {delay:.1f} seconds..."
            )

            await asyncio.sleep(delay)

    raise RuntimeError("Baseline retry loop ended unexpectedly.")


async def evaluate_split(
    split: str,
    model: str = BASELINE_DEFAULT_MODEL,
    rerun: bool = False,
) -> dict[str, int]:
    """Run the baseline and evaluate one dataset split."""

    samples = load_samples(split)

    # Resume skips successful examples for this split and model.
    # A fresh run is managed once by run_all_splits(), not here.
    completed = set() if rerun else load_completed_ids(split, model)

    counts = {
        "total": len(samples),
        "success": 0,
        "failed": 0,
        "skipped": 0,
    }

    for sample in samples:
        example_id = str(sample["example_id"])

        if example_id in completed:
            counts["skipped"] += 1
            continue

        record = {
            "split": split,
            "example_id": example_id,
            "source_file": sample["source_file"],
            "source_text": sample["source_text"],
            "model": model,
        }

        try:
            generated_text = await run_baseline_with_retry(
                sample["source_text"],
                model=model,
            )

            if not isinstance(generated_text, str) or not generated_text.strip():
                raise ValueError("The baseline returned empty text.")

            metrics = evaluate_single(generated_text)

            record.update(
                {
                    "generated_text": generated_text,
                    "metrics": metrics,
                    "status": "success",
                    "error": None,
                }
            )
            counts["success"] += 1

        except Exception as exc:
            record.update(
                {
                    "generated_text": None,
                    "metrics": None,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            counts["failed"] += 1

        save_record(record)

    return counts


async def run_all_splits(
    model: str = BASELINE_DEFAULT_MODEL,
    rerun: bool = False,
) -> dict[str, dict[str, int]]:
    """Evaluate all splits, optionally starting from a clean results file."""

    if rerun:
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_FILE.unlink(missing_ok=True)

    all_counts = {}

    for split in VALID_SPLITS:
        print(f"\nEvaluating {split} split...")

        counts = await evaluate_split(split, model=model)

        all_counts[split] = counts
        print(f"{split} results: {counts}")

    return all_counts


def run_evaluation(
    split: str,
    model: str = BASELINE_DEFAULT_MODEL,
    rerun: bool = False,
) -> dict[str, int]:
    """Run evaluation for one split."""

    return asyncio.run(evaluate_split(split, model=model, rerun=rerun))


if __name__ == "__main__":
    # Set rerun=True for a fresh evaluation of all splits.
    # Set rerun=False to resume and skip previously successful samples.
    asyncio.run(run_all_splits(rerun=False))
