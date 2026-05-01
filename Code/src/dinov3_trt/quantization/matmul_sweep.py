"""Utilities for MatMul-only transformer block INT8 sweep experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MATMUL_NODE_SUFFIXES: tuple[str, ...] = (
    "attention/q_proj/MatMul",
    "attention/k_proj/MatMul",
    "attention/v_proj/MatMul",
    "attention/MatMul",
    "attention/MatMul_1",
    "attention/o_proj/MatMul",
    "mlp/up_proj/MatMul",
    "mlp/down_proj/MatMul",
)

MATMUL_NODE_GROUP_SUFFIXES: dict[str, tuple[str, ...]] = {
    "all": MATMUL_NODE_SUFFIXES,
    "attention": MATMUL_NODE_SUFFIXES[:6],
    "qkv": MATMUL_NODE_SUFFIXES[:3],
    "attention-core": MATMUL_NODE_SUFFIXES[3:5],
    "attention-out": (MATMUL_NODE_SUFFIXES[5],),
    "mlp": MATMUL_NODE_SUFFIXES[6:],
    "mlp-up": (MATMUL_NODE_SUFFIXES[6],),
    "mlp-down": (MATMUL_NODE_SUFFIXES[7],),
}


@dataclass(frozen=True)
class MatMulSweepVariant:
    """One MatMul-only quantization variant."""

    blocks: tuple[int, ...]
    node_group: str = "all"

    def __post_init__(self) -> None:
        if self.node_group not in MATMUL_NODE_GROUP_SUFFIXES:
            valid = ", ".join(sorted(MATMUL_NODE_GROUP_SUFFIXES))
            raise ValueError(f"unknown MatMul node group: {self.node_group}; valid: {valid}")

    @property
    def label(self) -> str:
        block_label = format_block_label(self.blocks)
        if self.node_group == "all":
            return block_label
        return f"{block_label}_{self.node_group.replace('-', '_')}"

    @property
    def node_suffixes(self) -> tuple[str, ...]:
        return MATMUL_NODE_GROUP_SUFFIXES[self.node_group]

    @property
    def nodes_to_quantize(self) -> tuple[str, ...]:
        return matmul_nodes_for_blocks(self.blocks, suffixes=self.node_suffixes)


@dataclass(frozen=True)
class MatMulSweepPaths:
    """Artifact paths for one sweep variant."""

    quantized_onnx: Path
    quantize_stdout: Path
    random_compare: Path
    random_compare_stdout: Path
    image_eval: Path
    image_eval_stdout: Path


def parse_block_spec(value: str) -> tuple[int, ...]:
    """Parse comma-separated block indices and inclusive ranges."""

    blocks: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start = _parse_block_index(start_text)
            end = _parse_block_index(end_text)
            if end < start:
                raise ValueError(f"block range end must be >= start: {part}")
            blocks.extend(range(start, end + 1))
        else:
            blocks.append(_parse_block_index(part))
    if not blocks:
        raise ValueError("at least one block index is required")
    return tuple(sorted(set(blocks)))


def parse_variant_specs(
    values: tuple[str, ...],
    *,
    node_groups: tuple[str, ...] = ("all",),
) -> tuple[MatMulSweepVariant, ...]:
    """Parse CLI variant specs into unique sweep variants."""

    variants: list[MatMulSweepVariant] = []
    seen: set[tuple[tuple[int, ...], str]] = set()
    groups = tuple(parse_node_group(group) for group in node_groups)
    for value in values:
        blocks, inline_group = parse_variant_spec(value)
        selected_groups = (inline_group,) if inline_group is not None else groups
        for group in selected_groups:
            key = (blocks, group)
            if key in seen:
                continue
            seen.add(key)
            variants.append(MatMulSweepVariant(blocks=blocks, node_group=group))
    if not variants:
        raise ValueError("at least one sweep variant is required")
    return tuple(variants)


def parse_variant_spec(value: str) -> tuple[tuple[int, ...], str | None]:
    """Parse one variant as BLOCKS or BLOCKS:node-group."""

    block_text, separator, group_text = value.partition(":")
    blocks = parse_block_spec(block_text)
    if not separator:
        return blocks, None
    return blocks, parse_node_group(group_text)


def parse_node_group(value: str) -> str:
    """Validate a MatMul node group."""

    group = value.strip()
    if group not in MATMUL_NODE_GROUP_SUFFIXES:
        valid = ", ".join(sorted(MATMUL_NODE_GROUP_SUFFIXES))
        raise ValueError(f"unknown MatMul node group: {group}; valid: {valid}")
    return group


def format_block_label(blocks: tuple[int, ...]) -> str:
    """Return a filename-safe label for a block tuple."""

    if not blocks:
        raise ValueError("at least one block index is required")
    if len(blocks) == 1:
        return f"layer{blocks[0]}"
    expected = tuple(range(blocks[0], blocks[-1] + 1))
    if blocks == expected:
        return f"layers{blocks[0]}_{blocks[-1]}"
    return "layers" + "_".join(str(block) for block in blocks)


def matmul_nodes_for_blocks(
    blocks: tuple[int, ...],
    *,
    suffixes: tuple[str, ...] = MATMUL_NODE_SUFFIXES,
) -> tuple[str, ...]:
    """Return exact ONNX node names for MatMul nodes in the given transformer blocks."""

    for block in blocks:
        _validate_block_index(block)
    return tuple(
        f"/model/layer.{block}/{suffix}"
        for block in blocks
        for suffix in suffixes
    )


def make_sweep_paths(
    *,
    onnx_dir: Path,
    reports_dir: Path,
    model_stem: str,
    prefix: str,
    variant: MatMulSweepVariant,
    quantize_mode: str = "int8",
) -> MatMulSweepPaths:
    """Return standard artifact paths for one sweep variant.

    `quantize_mode` (currently ``"int8"`` or ``"fp8"``) is folded into the file
    names so an INT8 sweep and an FP8 sweep can coexist without overwriting
    each other's ONNX or ORT comparison reports.
    """

    label = variant.label
    report_stem = f"{prefix}_{label}"
    onnx_name = f"{model_stem}.{quantize_mode}.modelopt.{report_stem}.onnx"
    return MatMulSweepPaths(
        quantized_onnx=onnx_dir / onnx_name,
        quantize_stdout=reports_dir / f"quantize_modelopt_{report_stem}_{quantize_mode}.stdout.txt",
        random_compare=reports_dir
        / f"compare_onnx_fp32_vs_{quantize_mode}_modelopt_{report_stem}_b1.json",
        random_compare_stdout=reports_dir
        / f"compare_onnx_fp32_vs_{quantize_mode}_modelopt_{report_stem}_b1.stdout.txt",
        image_eval=reports_dir
        / f"eval_onnx_imagenette32_fp32_vs_{quantize_mode}_modelopt_{report_stem}.json",
        image_eval_stdout=reports_dir
        / f"eval_onnx_imagenette32_fp32_vs_{quantize_mode}_modelopt_{report_stem}.stdout.txt",
    )


def _parse_block_index(value: str) -> int:
    try:
        block = int(value.strip())
    except ValueError as exc:
        raise ValueError(f"block index must be an integer: {value}") from exc
    _validate_block_index(block)
    return block


def _validate_block_index(block: int) -> None:
    if block < 0 or block > 19:
        raise ValueError("DINOv3 ViT-L/16 exported block indices must be in 0..19")
