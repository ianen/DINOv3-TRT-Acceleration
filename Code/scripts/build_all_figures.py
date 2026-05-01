#!/usr/bin/env python
"""Regenerate every project figure (4 sub-systems) in one command.

Subsystems invoked, in order:
1. **Speedup bars** — `build_benchmark_figures(matrix_csv, output_dir)` reads
   `formal_benchmark_matrix.csv` and emits the trtexec/cpp speedup bar SVGs.
2. **Cosine bars** — `build_cosine_figures(reports_dir, output_dir)` reads the
   `eval_imagenette1000_*.json` reports and emits per-resolution cosine SVGs.
3. **Tradeoff scatter** — `build_tradeoff_figures(reports_dir, output_dir)` joins
   eval + speedup reports into the cosine-vs-speedup scatter SVG.
4. **Layer ablation** — `build_layer_ablation_figures(reports_dir, output_dir)`
   plots the 4-layer ablation diversity-vs-balance SVG.

`--allow-missing` is propagated to every subsystem; failures in one subsystem do
not abort the others when this flag is set.

The script writes a top-level `figures_index.json` summarising which subsystem
produced which output, with row counts and missing-report lists. Callers can
diff this index across runs to confirm that nothing regressed.
"""

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

from dinov3_trt.reports.benchmark_figures import (  # noqa: E402
    build_benchmark_figures,
    build_cosine_figures,
    build_layer_ablation_figures,
    build_tradeoff_figures,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-csv",
        type=Path,
        default=Path("Artifacts") / "reports" / "formal_benchmark_matrix.csv",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("Artifacts") / "reports",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Artifacts") / "reports" / "figures",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip figures whose source reports are missing instead of failing.",
    )
    parser.add_argument(
        "--index-output",
        type=Path,
        default=None,
        help="Override path for the top-level figures_index.json (default: <output-dir>/figures_index.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index: dict[str, Any] = {
        "matrix_csv": str(args.matrix_csv),
        "reports_dir": str(args.reports_dir),
        "output_dir": str(args.output_dir),
        "allow_missing": bool(args.allow_missing),
        "subsystems": {},
    }

    speedup_manifest = build_benchmark_figures(
        args.matrix_csv,
        args.output_dir,
        allow_missing=args.allow_missing,
    )
    index["subsystems"]["speedup"] = _summarise_manifest(speedup_manifest)

    cosine_manifest = build_cosine_figures(
        args.reports_dir,
        args.output_dir,
        allow_missing=args.allow_missing,
    )
    index["subsystems"]["cosine"] = _summarise_manifest(cosine_manifest)

    tradeoff_manifest = build_tradeoff_figures(
        args.reports_dir,
        args.output_dir,
        allow_missing=args.allow_missing,
    )
    index["subsystems"]["tradeoff"] = _summarise_manifest(tradeoff_manifest)

    layer_ablation_manifest = build_layer_ablation_figures(
        args.reports_dir,
        args.output_dir,
        allow_missing=args.allow_missing,
    )
    index["subsystems"]["layer_ablation"] = _summarise_manifest(layer_ablation_manifest)

    index_path = args.index_output or (args.output_dir / "figures_index.json")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(index, indent=2))
    print(f"\nfigures_index -> {index_path}")
    return 0


def _summarise_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    figures = manifest.get("figures") or []
    summary: list[dict[str, Any]] = []
    for figure in figures:
        if not isinstance(figure, dict):
            continue
        entry: dict[str, Any] = {
            "name": figure.get("name"),
            "row_count": figure.get("row_count"),
        }
        output = figure.get("output")
        if output is not None:
            entry["output"] = output
        missing_reports = figure.get("missing_reports")
        if missing_reports:
            entry["missing_reports"] = missing_reports
        missing_report = figure.get("missing_report")
        if missing_report:
            entry["missing_report"] = missing_report
        summary.append(entry)
    return {"figure_count": len(figures), "figures": summary}


if __name__ == "__main__":
    raise SystemExit(main())
