"""End-to-end post-download orchestrator for ImageNet val 50K cosine eval.

Triggers the V1.0.1 §12.1 final-item closure once the kagglehub-driven
ImageNet download (workaround for HF 403 GatedRepoError) completes.

Pipeline
--------
1. Resolve the downloaded dataset path from ``download.success`` marker
   (or accept ``--image-root`` directly for manual invocation).
2. Generate disjoint eval (default 1000) + calib (default 500) manifests
   under ``Artifacts/manifests/`` via the existing
   ``prepare_image_subset_manifests.py``.
3. Run cosine eval for each registered (reference, candidate) engine pair
   via ``evaluate_engine_pair_on_images.py`` and aggregate per-output
   cos_min / cos_mean / max_abs_error metrics.
4. Write a unified summary report listing every candidate's verdict
   against the V1.0.1 §12.1 cos thresholds (R1 ≥ 0.99 strict / R2 ≥ 0.97
   emergency).

Default candidates
------------------
- BF16 prefer (V1.0.1 main delivery candidate; expected cos_min ≥ 0.998)
- SmoothQuant α=0.8 INT8 (R2 emergency candidate; expected cos_min ~0.97)

Manual invocation example::

    .venv\\Scripts\\python.exe scripts\\run_imagenet_val_post_download.py \\
        --image-root D:\\path\\to\\extracted_imagenet_val \\
        --eval-count 1000 --calib-count 500 --seed 42

When the download.success marker is present, the script auto-resolves the
image root from it; ``--image-root`` is then optional.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_KAGGLEHUB_LOG_DIR = Path(
    "Artifacts/datasets/imagenet_val_kagglehub"
).resolve()
DEFAULT_MANIFEST_DIR = Path("Artifacts/manifests").resolve()
DEFAULT_REPORT_DIR = Path("Artifacts/reports").resolve()
DEFAULT_ENGINES_DIR = Path("Artifacts/engines").resolve()


@dataclass(frozen=True)
class EnginePair:
    """A reference/candidate engine pair to compare on the eval manifest."""

    label: str
    reference: str
    candidate: str
    cos_min_threshold_r1: float = 0.99
    cos_min_threshold_r2: float = 0.97


DEFAULT_PAIRS: tuple[EnginePair, ...] = (
    EnginePair(
        label="bf16_prefer",
        reference="dinov3_vitl16_4out.fp32.engine",
        candidate="dinov3_vitl16_4out.bf16.prefer.engine",
    ),
    EnginePair(
        label="int8_smoothquant_a080",
        reference="dinov3_vitl16_4out.fp32.engine",
        candidate=(
            "dinov3_vitl16_4out.int8.modelopt.smoothquant."
            "alpha080.imagenette500.engine"
        ),
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Path to the directory containing extracted ImageNet val JPEGs. "
            "If omitted, resolved from "
            "Artifacts/datasets/imagenet_val_kagglehub/download.success."
        ),
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help="Output directory for generated eval/calib manifests.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Output directory for cosine eval reports.",
    )
    parser.add_argument(
        "--engines-dir",
        type=Path,
        default=DEFAULT_ENGINES_DIR,
        help="Directory containing the .engine files referenced by the pairs.",
    )
    parser.add_argument(
        "--eval-count",
        type=int,
        default=1000,
        help="Number of images for the eval manifest.",
    )
    parser.add_argument(
        "--calib-count",
        type=int,
        default=500,
        help="Number of images for the calib manifest (disjoint from eval).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for manifest sampling."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size for cosine eval.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Inference resolution.",
    )
    parser.add_argument(
        "--skip-pair",
        action="append",
        default=[],
        choices=[p.label for p in DEFAULT_PAIRS],
        help=(
            "Pair label to skip (repeatable). Default runs every registered "
            "pair."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve paths and print the planned commands without executing.",
    )
    return parser.parse_args(argv)


def resolve_image_root(
    cli_image_root: Path | None,
    *,
    success_marker: Path = DEFAULT_KAGGLEHUB_LOG_DIR / "download.success",
) -> Path:
    """Resolve the dataset image root.

    Priority: ``--image-root`` flag → ``download.success`` marker contents.
    Walks one level into the kagglehub cache when the marker points at the
    versions directory (kagglehub returns the version folder, but actual
    images may be one or two levels deeper depending on dataset packaging).
    """

    if cli_image_root is not None:
        if not cli_image_root.is_dir():
            raise SystemExit(f"--image-root '{cli_image_root}' does not exist")
        return cli_image_root.resolve()

    if not success_marker.is_file():
        raise SystemExit(
            f"download.success marker missing at '{success_marker}'. "
            "Either wait for the kagglehub download to complete or pass "
            "--image-root explicitly."
        )

    raw = success_marker.read_text(encoding="utf-8").strip()
    candidate = Path(raw)
    if not candidate.is_dir():
        raise SystemExit(
            f"path inside download.success ('{candidate}') is not a directory"
        )

    # When the path returned by kagglehub points at a versions/ root, descend
    # into the first non-empty subdirectory that contains any image files.
    if not _has_images_directly(candidate):
        for child in sorted(candidate.iterdir()):
            if child.is_dir() and _has_images_anywhere(child):
                candidate = child
                break

    return candidate.resolve()


def _has_images_directly(directory: Path) -> bool:
    return any(
        p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in directory.iterdir()
    )


def _has_images_anywhere(directory: Path) -> bool:
    return any(
        p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in directory.rglob("*")
    )


def build_manifest_paths(manifest_dir: Path, eval_count: int) -> tuple[Path, Path]:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    eval_manifest = manifest_dir / f"imagenet_val_50k_eval_{eval_count}.json"
    calib_manifest = manifest_dir / f"imagenet_val_50k_calib_{eval_count}.json"
    return eval_manifest, calib_manifest


def run_subprocess(cmd: list[str], *, dry_run: bool) -> int:
    print(f"[run] {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def generate_manifests(
    *,
    image_root: Path,
    eval_manifest: Path,
    calib_manifest: Path,
    eval_count: int,
    calib_count: int,
    seed: int,
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        "scripts/prepare_image_subset_manifests.py",
        "--image-root",
        str(image_root),
        "--eval-output",
        str(eval_manifest),
        "--calib-output",
        str(calib_manifest),
        "--eval-count",
        str(eval_count),
        "--calib-count",
        str(calib_count),
        "--seed",
        str(seed),
    ]
    rc = run_subprocess(cmd, dry_run=dry_run)
    if rc != 0:
        raise SystemExit(f"manifest generation failed (rc={rc})")


def run_cosine_eval(
    *,
    pair: EnginePair,
    engines_dir: Path,
    manifest: Path,
    report_dir: Path,
    batch_size: int,
    image_size: int,
    dry_run: bool,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / f"eval_imagenet50k_{pair.label}.json"
    reference_engine = engines_dir / pair.reference
    candidate_engine = engines_dir / pair.candidate

    if not reference_engine.exists():
        raise SystemExit(f"reference engine missing: {reference_engine}")
    if not candidate_engine.exists():
        raise SystemExit(f"candidate engine missing: {candidate_engine}")

    cmd = [
        sys.executable,
        "scripts/evaluate_engine_pair_on_images.py",
        "--reference-engine",
        str(reference_engine),
        "--candidate-engine",
        str(candidate_engine),
        "--manifest",
        str(manifest),
        "--batch-size",
        str(batch_size),
        "--image-size",
        str(image_size),
        "--output",
        str(output),
    ]
    rc = run_subprocess(cmd, dry_run=dry_run)
    if rc != 0:
        raise SystemExit(f"cosine eval failed for pair '{pair.label}' (rc={rc})")
    return output


_COS_MIN_KEYS: tuple[str, ...] = ("cosine_similarity_min", "cosine_min", "cos_min")
_COS_MEAN_KEYS: tuple[str, ...] = (
    "cosine_similarity_mean",
    "cosine_mean",
    "cos_mean",
)


def _first_present(metrics: dict[str, object], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        value = metrics.get(k)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _normalize_outputs(payload: dict[str, object]) -> list[dict[str, object]]:
    """Return per-output metric dicts as a list, regardless of source layout.

    Supports the canonical ``evaluate_engine_pair_on_images.py`` schema
    (``outputs: list[dict]`` with ``name``) and a hypothetical mapping
    layout (``per_output_metrics: dict[name, metrics]``) used by older
    fixtures.
    """
    outputs: object = payload.get("outputs")
    if isinstance(outputs, list):
        return [o for o in outputs if isinstance(o, dict)]
    legacy: object = payload.get("per_output_metrics")
    if isinstance(legacy, dict):
        return [
            {"name": name, **metrics}
            for name, metrics in legacy.items()
            if isinstance(metrics, dict)
        ]
    if isinstance(legacy, list):
        return [o for o in legacy if isinstance(o, dict)]
    return []


def summarize_pair(
    pair: EnginePair, report_path: Path
) -> dict[str, object]:
    """Extract per-output cos_min / cos_mean and apply R1/R2 verdict."""

    if not report_path.exists():
        return {"label": pair.label, "verdict": "REPORT_MISSING"}
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    outputs = _normalize_outputs(payload)
    cos_min_global = float("inf")
    cos_mean_total = 0.0
    cos_mean_count = 0
    output_summaries: list[dict[str, object]] = []

    for o in outputs:
        name = str(o.get("name", "?"))
        cm = _first_present(o, _COS_MIN_KEYS)
        cmu = _first_present(o, _COS_MEAN_KEYS)
        if cm is not None:
            cos_min_global = min(cos_min_global, cm)
        if cmu is not None:
            cos_mean_total += cmu
            cos_mean_count += 1
        output_summaries.append({"output": name, "cos_min": cm, "cos_mean": cmu})

    if cos_min_global == float("inf"):
        cos_min_global = -1.0
    avg_mean = cos_mean_total / cos_mean_count if cos_mean_count else -1.0

    if cos_min_global >= pair.cos_min_threshold_r1:
        verdict = "R1_PASS_strict"
    elif cos_min_global >= pair.cos_min_threshold_r2:
        verdict = "R2_PASS_emergency"
    else:
        verdict = "FAIL"

    return {
        "label": pair.label,
        "report_path": str(report_path),
        "cos_min_overall": cos_min_global,
        "cos_mean_overall": avg_mean,
        "verdict": verdict,
        "per_output": output_summaries,
        "thresholds": {
            "R1_strict": pair.cos_min_threshold_r1,
            "R2_emergency": pair.cos_min_threshold_r2,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    image_root = resolve_image_root(args.image_root)
    print(f"[image_root] {image_root}", flush=True)

    eval_manifest, calib_manifest = build_manifest_paths(
        args.manifest_dir, args.eval_count
    )
    print(f"[manifests] eval={eval_manifest} calib={calib_manifest}", flush=True)

    generate_manifests(
        image_root=image_root,
        eval_manifest=eval_manifest,
        calib_manifest=calib_manifest,
        eval_count=args.eval_count,
        calib_count=args.calib_count,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    pairs = [p for p in DEFAULT_PAIRS if p.label not in set(args.skip_pair)]
    if not pairs:
        raise SystemExit("no engine pairs to run after applying --skip-pair filters")

    summaries: list[dict[str, object]] = []
    for pair in pairs:
        print(f"\n[pair] {pair.label}", flush=True)
        report = run_cosine_eval(
            pair=pair,
            engines_dir=args.engines_dir,
            manifest=eval_manifest,
            report_dir=args.report_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            dry_run=args.dry_run,
        )
        summaries.append(summarize_pair(pair, report))

    summary_path = args.report_dir / "imagenet50k_post_download_summary.json"
    summary_payload = {
        "image_root": str(image_root),
        "eval_manifest": str(eval_manifest),
        "calib_manifest": str(calib_manifest),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pairs": summaries,
    }
    if not args.dry_run:
        summary_path.write_text(
            json.dumps(summary_payload, indent=2), encoding="utf-8"
        )
        print(f"\n[summary] wrote {summary_path}", flush=True)

    print("\n=== V1.0.1 §12.1 Final-Item Verdict ===", flush=True)
    for s in summaries:
        cm = s.get("cos_min_overall", -1.0)
        verdict = s.get("verdict", "?")
        print(f"  {s['label']:30s} cos_min={cm:.4f} -> {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
