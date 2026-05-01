#!/usr/bin/env python
"""4-layer-combination ablation for DINOv3 ViT-L/16 (V1.1 stretch goal #3).

Run a single PyTorch HF DINOv3 forward per batch, capture all 24 transformer
block hidden states with `output_hidden_states=True`, then for each candidate
layer combination compute:

* **Pairwise cosine similarity** between the 4 selected outputs (per-image
  flattened over `[1+patch, hidden]`, averaged across the dataset). A *lower*
  mean pairwise cosine indicates the 4 hooks expose more diverse multi-scale
  features — the property DPT-style fusion heads rely on.
* **Per-output magnitude** (mean L2 norm across `[B,T*C]`) — sanity-checks that
  no candidate position produces drastically smaller activations than the rest.

Designed to be cheap: one forward per batch covers all candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.contracts import (  # noqa: E402
    DINO_VITL16_LAYER_ABLATION_CANDIDATES,
    DINO_VITL16_NUM_BLOCKS,
    make_dinov3_vitl16_contract,
)
from dinov3_trt.infer.image_eval import (  # noqa: E402
    chunk_paths,
    list_image_paths,
    load_image_batch,
    read_image_manifest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True, type=str)
    parser.add_argument("--manifest", type=Path, help="JSON image manifest")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-md", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--candidates",
        default="project,dpt,late",
        help="CSV of candidate names from contracts.DINO_VITL16_LAYER_ABLATION_CANDIDATES",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip torch import, validate args + write a plan JSON.",
    )
    return parser.parse_args(argv)


def select_candidates(names_csv: str) -> dict[str, tuple[int, ...]]:
    """Resolve a CSV name list against the registered ablation candidates."""

    names = [n.strip() for n in names_csv.split(",") if n.strip()]
    if not names:
        raise ValueError("--candidates must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError("--candidates must not contain duplicates")
    selected: dict[str, tuple[int, ...]] = {}
    for name in names:
        if name not in DINO_VITL16_LAYER_ABLATION_CANDIDATES:
            raise ValueError(
                f"unknown candidate '{name}'. "
                f"Available: {sorted(DINO_VITL16_LAYER_ABLATION_CANDIDATES)}"
            )
        selected[name] = DINO_VITL16_LAYER_ABLATION_CANDIDATES[name]
    return selected


def pairwise_cosine_per_batch(features: np.ndarray) -> np.ndarray:
    """`features` shape `[N, B, T*C]`; returns mean cosine over `B` for each pair."""

    if features.ndim != 3:
        raise ValueError("features must have shape [N, B, T*C]")
    n_outputs = features.shape[0]
    norm = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-12)
    pairs: list[float] = []
    for i in range(n_outputs):
        for j in range(i + 1, n_outputs):
            cos_per_sample = np.sum(norm[i] * norm[j], axis=-1)
            pairs.append(float(cos_per_sample.mean()))
    return np.asarray(pairs, dtype=np.float64)


def per_output_magnitude(features: np.ndarray) -> np.ndarray:
    """`features` shape `[N, B, T*C]`; returns mean L2 over `B` per output -> `[N]`."""

    if features.ndim != 3:
        raise ValueError("features must have shape [N, B, T*C]")
    norms = np.linalg.norm(features, axis=-1)
    return np.asarray(norms.mean(axis=-1), dtype=np.float64)


def pair_labels(layer_indices_one_based: tuple[int, ...]) -> list[str]:
    labels: list[str] = []
    for i in range(len(layer_indices_one_based)):
        for j in range(i + 1, len(layer_indices_one_based)):
            labels.append(f"L{layer_indices_one_based[i]}-L{layer_indices_one_based[j]}")
    return labels


def write_dry_run_plan(args: argparse.Namespace, candidates: dict[str, tuple[int, ...]]) -> None:
    plan: dict[str, object] = {
        "dry_run": True,
        "model_name_or_path": args.model_name_or_path,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "max_images": args.max_images,
        "candidates": {name: list(indices) for name, indices in candidates.items()},
        "candidates_one_based": {
            name: [i + 1 for i in indices] for name, indices in candidates.items()
        },
        "num_blocks": DINO_VITL16_NUM_BLOCKS,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def aggregate_summary(
    candidates: dict[str, tuple[int, ...]],
    cosine_pair_lists: Mapping[str, list[list[float]]],
    magnitude_lists: Mapping[str, list[list[float]]],
    image_size: int,
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for name, indices in candidates.items():
        contract = make_dinov3_vitl16_contract(image_size, layer_indices=indices)
        per_batch_cos = np.asarray(cosine_pair_lists[name])
        per_batch_mag = np.asarray(magnitude_lists[name])
        summary[name] = {
            "layer_indices_0based": list(indices),
            "layer_numbers_1based": list(contract.layer_numbers),
            "output_names": list(contract.output_names),
            "pairwise_cosine_labels": pair_labels(contract.layer_numbers),
            "pairwise_cosine_per_pair_mean": per_batch_cos.mean(axis=0).tolist(),
            "pairwise_cosine_overall_mean": float(per_batch_cos.mean()),
            "pairwise_cosine_overall_min": float(per_batch_cos.min()),
            "pairwise_cosine_overall_max": float(per_batch_cos.max()),
            "per_output_magnitude_mean": per_batch_mag.mean(axis=0).tolist(),
            "per_output_magnitude_std": per_batch_mag.std(axis=0).tolist(),
        }
    return summary


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = [
        "# 4 层组合 ablation (V1.1 stretch goal #3)",
        "",
        f"- 模型: `{payload['model_name_or_path']}`",
        f"- image_size: {payload['image_size']}",
        f"- batch_size: {payload['batch_size']}",
        f"- image_count: {payload['image_count']}",
        "",
        "## Inter-output cosine (越低 = 输出越异质，对多尺度融合越有益)",
        "",
        "| candidate | layers (1-based) | overall mean cos | overall min cos | overall max cos |",
        "|---|---|---:|---:|---:|",
    ]
    for name, summary in payload["candidates"].items():
        layers = "/".join(str(n) for n in summary["layer_numbers_1based"])
        lines.append(
            f"| {name} | {layers} | {summary['pairwise_cosine_overall_mean']:.4f}"
            f" | {summary['pairwise_cosine_overall_min']:.4f}"
            f" | {summary['pairwise_cosine_overall_max']:.4f} |"
        )
    lines.append("")
    lines.append("## Per-output magnitude (mean L2)")
    lines.append("")
    lines.append("| candidate | layer | magnitude mean | magnitude std |")
    lines.append("|---|---|---:|---:|")
    for name, summary in payload["candidates"].items():
        for layer, mag, mstd in zip(
            summary["layer_numbers_1based"],
            summary["per_output_magnitude_mean"],
            summary["per_output_magnitude_std"],
        ):
            lines.append(f"| {name} | L{layer} | {mag:.2f} | {mstd:.2f} |")
    lines.append("")
    lines.append("## Diversity ranking (mean cos 升序)")
    lines.append("")
    for rank_index, name in enumerate(payload["diversity_ranking_low_to_high_cosine"], start=1):
        lines.append(f"{rank_index}. **{name}**")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidates = select_candidates(args.candidates)
    if args.dry_run:
        write_dry_run_plan(args, candidates)
        print(f"[dry-run] plan written to {args.output}")
        return 0

    if (args.image_root is None) == (args.manifest is None):
        raise SystemExit("provide exactly one of --image-root or --manifest")
    if args.manifest is not None:
        image_paths = read_image_manifest(args.manifest)
    else:
        image_paths = list_image_paths(args.image_root, recursive=True)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise SystemExit("no images selected")

    import torch  # type: ignore[import-not-found]  # noqa: PLC0415

    from dinov3_trt.export.hf_model import (  # noqa: PLC0415
        create_hf_dinov3_model,
        drop_register_tokens,
    )

    model = create_hf_dinov3_model(args.model_name_or_path)
    model.eval()
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)

    config = getattr(model, "config", None)
    num_register_tokens = int(getattr(config, "num_register_tokens", 0))
    if num_register_tokens < 0:
        raise SystemExit("model.config.num_register_tokens must be >= 0")

    cosine_pair_lists: dict[str, list[list[float]]] = {name: [] for name in candidates}
    magnitude_lists: dict[str, list[list[float]]] = {name: [] for name in candidates}
    image_count_seen = 0

    for path_batch in chunk_paths(image_paths, args.batch_size):
        image_batch = load_image_batch(path_batch, image_size=args.image_size)
        with torch.no_grad():
            tensor = torch.from_numpy(image_batch.tensor).to(device)
            outputs = model(pixel_values=tensor, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise SystemExit("model did not return hidden_states (check transformers version)")
        if len(hidden_states) <= DINO_VITL16_NUM_BLOCKS:
            raise SystemExit(
                f"expected {DINO_VITL16_NUM_BLOCKS + 1} hidden states (1 emb + 24 blocks), "
                f"got {len(hidden_states)}"
            )

        for name, indices in candidates.items():
            stacked: list[np.ndarray] = []
            for layer_index in indices:
                hs = hidden_states[layer_index + 1]
                hs = drop_register_tokens(hs, num_register_tokens)
                stacked.append(hs.detach().cpu().numpy())
            arr = np.stack([s.reshape(s.shape[0], -1) for s in stacked], axis=0)
            pair_cos = pairwise_cosine_per_batch(arr)
            cosine_pair_lists[name].append(pair_cos.tolist())
            magnitude_lists[name].append(per_output_magnitude(arr).tolist())

        image_count_seen += len(path_batch)
        if image_count_seen % 100 == 0 or image_count_seen == len(image_paths):
            print(f"  processed {image_count_seen}/{len(image_paths)} images")

    summary = aggregate_summary(
        candidates,
        cosine_pair_lists,
        magnitude_lists,
        args.image_size,
    )
    diversity_ranking = sorted(
        summary,
        key=lambda k: float(summary[k]["pairwise_cosine_overall_mean"]),  # type: ignore[arg-type]
    )
    payload: dict[str, Any] = {
        "model_name_or_path": args.model_name_or_path,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "image_count": image_count_seen,
        "candidates": summary,
        "diversity_ranking_low_to_high_cosine": diversity_ranking,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote summary -> {args.output}")

    if args.report_md is not None:
        write_markdown_report(args.report_md, payload)
        print(f"wrote markdown -> {args.report_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
