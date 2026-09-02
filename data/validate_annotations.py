#!/usr/bin/env python3
"""Validate FaithfulMed MTSamples attribution annotations.

Uses only the Python standard library. The JSON Schema in annotation_templates/
is the portable format contract; this script additionally performs checks that
JSON Schema cannot express, such as filename, source-span, pair, and split rules.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CONTENT_TYPES = {"lab_values", "medications", "procedures", "instructions"}
FACT_TYPES = {"diagnosis", "medication", "lab_value", "procedure", "instruction"}
SPLITS = {"scratch", "dev", "test"}
SELECTION_MODES = {"entire_report", "self_contained_passage"}

COMMON_SOURCE_FIELDS = {
    "example_id",
    "source_url",
    "website_specialty",
    "split",
    "primary_content_type",
    "secondary_content_types",
    "selection_mode",
    "source_text",
}
INDEPENDENT_FIELDS = COMMON_SOURCE_FIELDS | {
    "annotator_id",
    "facts",
    "annotation_notes",
    "needs_adjudication",
}
ADJUDICATED_FIELDS = COMMON_SOURCE_FIELDS | {
    "source_dataset",
    "parent_document_id",
    "included_sections",
    "independent_annotators",
    "adjudicator_id",
    "annotation_version",
    "frozen",
    "gold_facts",
    "adjudication_notes",
}
FACT_FIELDS = {"fact_id", "fact_text", "fact_type", "source_sentence"}


@dataclass(frozen=True)
class Record:
    path: Path
    data: dict[str, Any]
    kind: str


def normalize_whitespace(value: str) -> str:
    """Collapse all Unicode whitespace for harmless source-span comparison."""

    return " ".join(value.split())


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def add_error(errors: list[str], path: Path, message: str) -> None:
    errors.append(f"{display_path(path)}: {message}")


def require_fields(
    data: dict[str, Any], allowed: set[str], path: Path, errors: list[str]
) -> None:
    missing = sorted(allowed - data.keys())
    extra = sorted(data.keys() - allowed)
    if missing:
        add_error(errors, path, f"missing required field(s): {', '.join(missing)}")
    if extra:
        add_error(errors, path, f"unexpected field(s): {', '.join(extra)}")


def require_nonempty_string(
    data: dict[str, Any], field: str, path: Path, errors: list[str]
) -> None:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        add_error(errors, path, f"{field} must be a nonempty string")


def require_string(
    data: dict[str, Any], field: str, path: Path, errors: list[str]
) -> None:
    if not isinstance(data.get(field), str):
        add_error(errors, path, f"{field} must be a string")


def require_enum(
    data: dict[str, Any],
    field: str,
    allowed: set[str],
    path: Path,
    errors: list[str],
) -> None:
    value = data.get(field)
    if not isinstance(value, str) or value not in allowed:
        values = ", ".join(sorted(allowed))
        add_error(errors, path, f"{field} must be one of: {values}")


def validate_string_array(
    data: dict[str, Any],
    field: str,
    path: Path,
    errors: list[str],
    *,
    allowed: set[str] | None = None,
    minimum: int = 0,
) -> None:
    value = data.get(field)
    if not isinstance(value, list):
        add_error(errors, path, f"{field} must be an array")
        return
    if len(value) < minimum:
        add_error(errors, path, f"{field} must contain at least {minimum} item(s)")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        add_error(errors, path, f"{field} items must be nonempty strings")
        return
    if len(value) != len(set(value)):
        add_error(errors, path, f"{field} must not contain duplicates")
    if allowed is not None:
        invalid = sorted(set(value) - allowed)
        if invalid:
            add_error(errors, path, f"{field} has invalid value(s): {', '.join(invalid)}")


def validate_facts(
    data: dict[str, Any], field: str, path: Path, errors: list[str], minimum: int
) -> None:
    facts = data.get(field)
    if not isinstance(facts, list):
        add_error(errors, path, f"{field} must be an array")
        return
    if len(facts) < minimum:
        add_error(errors, path, f"{field} must contain at least {minimum} fact(s)")

    seen_ids: set[str] = set()
    source_text = data.get("source_text")
    normalized_source = (
        normalize_whitespace(source_text) if isinstance(source_text, str) else ""
    )
    for index, fact in enumerate(facts):
        label = f"{field}[{index}]"
        if not isinstance(fact, dict):
            add_error(errors, path, f"{label} must be an object")
            continue
        missing = sorted(FACT_FIELDS - fact.keys())
        extra = sorted(fact.keys() - FACT_FIELDS)
        if missing:
            add_error(errors, path, f"{label} missing field(s): {', '.join(missing)}")
        if extra:
            add_error(errors, path, f"{label} has unexpected field(s): {', '.join(extra)}")

        for key in ("fact_id", "fact_text", "source_sentence"):
            value = fact.get(key)
            if not isinstance(value, str) or not value.strip():
                add_error(errors, path, f"{label}.{key} must be a nonempty string")

        fact_id = fact.get("fact_id")
        if isinstance(fact_id, str) and fact_id:
            if fact_id in seen_ids:
                add_error(errors, path, f"duplicate fact_id: {fact_id}")
            seen_ids.add(fact_id)

        if fact.get("fact_type") not in FACT_TYPES:
            values = ", ".join(sorted(FACT_TYPES))
            add_error(errors, path, f"{label}.fact_type must be one of: {values}")

        source_sentence = fact.get("source_sentence")
        if isinstance(source_sentence, str) and source_sentence.strip():
            normalized_sentence = normalize_whitespace(source_sentence)
            if normalized_sentence not in normalized_source:
                add_error(
                    errors,
                    path,
                    f"{label}.source_sentence does not appear in source_text "
                    "after whitespace normalization",
                )


def classify(data: dict[str, Any], path: Path, errors: list[str]) -> str | None:
    has_facts = "facts" in data
    has_gold = "gold_facts" in data
    if has_facts == has_gold:
        add_error(
            errors,
            path,
            "must contain exactly one of facts (independent) or gold_facts (adjudicated)",
        )
        return None
    return "independent" if has_facts else "adjudicated"


def validate_source_metadata(
    data: dict[str, Any], kind: str, path: Path, errors: list[str]
) -> None:
    for field in (
        "example_id",
        "source_url",
        "website_specialty",
        "source_text",
    ):
        require_nonempty_string(data, field, path, errors)
    require_enum(data, "split", SPLITS, path, errors)
    require_enum(data, "primary_content_type", CONTENT_TYPES, path, errors)
    validate_string_array(
        data, "secondary_content_types", path, errors, allowed=CONTENT_TYPES
    )
    require_enum(data, "selection_mode", SELECTION_MODES, path, errors)
    if kind == "adjudicated":
        require_nonempty_string(data, "parent_document_id", path, errors)
        if data.get("source_dataset") != "MTSamples":
            add_error(errors, path, 'source_dataset must equal "MTSamples"')
        validate_string_array(data, "included_sections", path, errors)


def validate_location(record: Record, errors: list[str]) -> None:
    split = record.data.get("split")
    if not isinstance(split, str) or split not in SPLITS:
        return

    parent = record.path.parent
    if split == "scratch":
        if parent.name != "scratch":
            add_error(errors, record.path, "scratch annotation must be directly in scratch/")
        if record.kind != "independent":
            add_error(errors, record.path, "scratch/ accepts independent annotations only")
        return

    expected_leaf = "independent" if record.kind == "independent" else "adjudicated"
    if parent.name != expected_leaf or parent.parent.name != split:
        add_error(
            errors,
            record.path,
            f"{record.kind} {split} annotation must be in {split}/{expected_leaf}/",
        )


def validate_record(record: Record, errors: list[str]) -> None:
    data, path, kind = record.data, record.path, record.kind
    allowed = INDEPENDENT_FIELDS if kind == "independent" else ADJUDICATED_FIELDS
    require_fields(data, allowed, path, errors)
    validate_source_metadata(data, kind, path, errors)

    example_id = data.get("example_id")
    if kind == "independent":
        require_nonempty_string(data, "annotator_id", path, errors)
        require_string(data, "annotation_notes", path, errors)
        if not isinstance(data.get("needs_adjudication"), bool):
            add_error(errors, path, "needs_adjudication must be a boolean")
        validate_facts(data, "facts", path, errors, 1)
        annotator_id = data.get("annotator_id")
        if isinstance(example_id, str) and isinstance(annotator_id, str):
            expected_name = f"{example_id}_{annotator_id}.json"
            if path.name != expected_name:
                add_error(errors, path, f"independent filename must be {expected_name}")
    else:
        validate_string_array(
            data, "independent_annotators", path, errors, minimum=2
        )
        require_nonempty_string(data, "adjudicator_id", path, errors)
        require_string(data, "adjudication_notes", path, errors)
        version = data.get("annotation_version")
        if not isinstance(version, str) or re.fullmatch(r"[0-9]+\.[0-9]+", version) is None:
            add_error(errors, path, "annotation_version must use major.minor format")
        if not isinstance(data.get("frozen"), bool):
            add_error(errors, path, "frozen must be a boolean")
        validate_facts(data, "gold_facts", path, errors, 1)
        if isinstance(example_id, str):
            expected_name = f"{example_id}_gold.json"
            if path.name != expected_name:
                add_error(errors, path, f"adjudicated filename must be {expected_name}")
        if data.get("split") == "test" and data.get("frozen") is not True:
            add_error(errors, path, "adjudicated test annotation must have frozen=true")

    validate_location(record, errors)


def load_records(paths: Iterable[Path], errors: list[str]) -> list[Record]:
    records: list[Record] = []
    for path in sorted(set(paths)):
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            add_error(errors, path, f"invalid JSON: {exc}")
            continue
        if not isinstance(data, dict):
            add_error(errors, path, "top-level JSON value must be an object")
            continue
        kind = classify(data, path, errors)
        if kind is not None:
            records.append(Record(path=path, data=data, kind=kind))
    return records


def json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".json" else []
    return list(path.rglob("*.json"))


def find_annotations_root(path: Path) -> Path | None:
    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        if candidate.name == "annotations":
            return candidate
    repository_default = Path(__file__).resolve().parent / "annotations"
    return repository_default if repository_default.exists() else None


def check_independent_pairs(records: list[Record], errors: list[str]) -> None:
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for record in records:
        data = record.data
        if record.kind == "independent":
            split = data.get("split")
            example_id = data.get("example_id")
            if (
                isinstance(split, str)
                and split in {"dev", "test"}
                and isinstance(example_id, str)
            ):
                groups[(split, example_id)].append(record)

    for (split, example_id), group in sorted(groups.items()):
        annotators = {
            record.data.get("annotator_id")
            for record in group
            if isinstance(record.data.get("annotator_id"), str)
        }
        if len(annotators) < 2:
            add_error(
                errors,
                group[0].path,
                f"{split} example {example_id} needs two distinct "
                "independent annotators",
            )


def check_parent_split_leakage(records: list[Record], errors: list[str]) -> None:
    parents_by_split: dict[str, set[str]] = {"dev": set(), "test": set()}
    path_by_parent: dict[tuple[str, str], Path] = {}
    for record in records:
        split = record.data.get("split")
        parent_id = record.data.get("parent_document_id")
        if (
            isinstance(split, str)
            and split in parents_by_split
            and isinstance(parent_id, str)
            and parent_id
        ):
            parents_by_split[split].add(parent_id)
            path_by_parent.setdefault((split, parent_id), record.path)

    for parent_id in sorted(parents_by_split["dev"] & parents_by_split["test"]):
        path = path_by_parent[("test", parent_id)]
        add_error(
            errors,
            path,
            f"parent_document_id {parent_id!r} appears in both dev and test",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate FaithfulMed MTSamples annotation JSON files."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "annotations"),
        help="JSON file or directory to validate (default: data/annotations)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = Path(args.path).resolve()
    if not target.exists():
        print(f"ERROR: path does not exist: {args.path}", file=sys.stderr)
        return 2

    selected_paths = json_files(target)
    errors: list[str] = []
    selected_records = load_records(selected_paths, errors)
    for record in selected_records:
        validate_record(record, errors)
    check_independent_pairs(selected_records, errors)

    # Leakage needs both sets even when the user validates only dev or test.
    annotations_root = find_annotations_root(target)
    if annotations_root is not None:
        global_paths = json_files(annotations_root / "dev") + json_files(
            annotations_root / "test"
        )
        global_errors: list[str] = []
        global_records = load_records(global_paths, global_errors)
        errors.extend(global_errors)
        check_parent_split_leakage(global_records, errors)

    if errors:
        for error in sorted(set(errors)):
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"Validation failed with {len(set(errors))} error(s).", file=sys.stderr)
        return 1

    print(f"Validated {len(selected_paths)} annotation file(s): no errors found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
