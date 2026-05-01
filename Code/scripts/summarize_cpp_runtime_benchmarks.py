#!/usr/bin/env python
"""Summarize paired C++ TensorRT runtime benchmark JSON reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.benchmarks.summary import (  # noqa: E402
    render_speedup_markdown,
    summarize_cpp_runtime_pair,
)


def _load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"report must be a JSON object: {path}")
    return data


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", required=True, type=Path)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    summary = summarize_cpp_runtime_pair(
        _load_report(args.reference_report),
        _load_report(args.candidate_report),
        reference_label=args.reference_label,
        candidate_label=args.candidate_label,
    )
    markdown = render_speedup_markdown(summary)

    if args.output_json is not None:
        _write_text(args.output_json, json.dumps(summary, indent=2) + "\n")
    if args.output_md is not None:
        _write_text(args.output_md, markdown + "\n")

    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
