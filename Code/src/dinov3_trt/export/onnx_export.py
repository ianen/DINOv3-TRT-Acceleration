"""ONNX export helpers for the 4-output DINOv3 contract."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dinov3_trt.contracts import DINO_VITL16_224_CONTRACT, ModelContract
from dinov3_trt.export.rope_patch import PatchReport, apply_dinov3_export_patches
from dinov3_trt.export.wrapper import DinoV3IntermediateLayerWrapper


@dataclass(frozen=True)
class OnnxExportConfig:
    output_path: Path
    opset: int = 18
    input_name: str = "pixel_values"
    batch_size: int = 1
    device: str = "cuda"
    dtype: str = "float32"
    dynamic_batch: bool = True
    apply_rope_patch: bool = True
    do_constant_folding: bool = True
    dynamo: bool = False

    def validate(self) -> None:
        if self.opset < 18:
            raise ValueError("DINOv3 export requires ONNX opset >= 18")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.dtype not in {"float32", "float16"}:
            raise ValueError("dtype must be float32 or float16")


def build_dynamic_axes(
    config: OnnxExportConfig,
    contract: ModelContract = DINO_VITL16_224_CONTRACT,
) -> dict[str, dict[int, str]] | None:
    if not config.dynamic_batch:
        return None
    return {
        config.input_name: {0: "batch"},
        **{name: {0: "batch"} for name in contract.output_names},
    }


def make_dummy_input(
    config: OnnxExportConfig,
    contract: ModelContract = DINO_VITL16_224_CONTRACT,
) -> Any:
    torch = importlib.import_module("torch")
    dtype = torch.float32 if config.dtype == "float32" else torch.float16
    return torch.zeros(
        (
            config.batch_size,
            3,
            contract.image_size,
            contract.image_size,
        ),
        dtype=dtype,
        device=config.device,
    )


@dataclass(frozen=True)
class OnnxExportResult:
    output_path: Path
    output_names: tuple[str, ...]
    dynamic_axes: dict[str, dict[int, str]] | None
    patch_report: PatchReport | None

    def to_json(self) -> dict[str, object]:
        return {
            "output_path": str(self.output_path),
            "output_names": list(self.output_names),
            "dynamic_axes": self.dynamic_axes,
            "patch_report": None if self.patch_report is None else self.patch_report.to_json(),
        }


def export_model_to_onnx(
    model: Any,
    config: OnnxExportConfig,
    contract: ModelContract = DINO_VITL16_224_CONTRACT,
) -> OnnxExportResult:
    config.validate()
    torch = importlib.import_module("torch")
    patch_report = apply_dinov3_export_patches() if config.apply_rope_patch else None
    wrapper = _make_torch_export_module(model, contract).eval()
    dummy_input = make_dummy_input(config, contract)
    dynamic_axes = build_dynamic_axes(config, contract)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            config.output_path,
            input_names=[config.input_name],
            output_names=list(contract.output_names),
            dynamic_axes=dynamic_axes,
            opset_version=config.opset,
            do_constant_folding=config.do_constant_folding,
            dynamo=config.dynamo,
        )

    return OnnxExportResult(
        output_path=config.output_path,
        output_names=contract.output_names,
        dynamic_axes=dynamic_axes,
        patch_report=patch_report,
    )


def _make_torch_export_module(model: Any, contract: ModelContract) -> Any:
    torch = importlib.import_module("torch")

    class TorchDinoV3IntermediateLayerWrapper(torch.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self, wrapped_model: Any, wrapped_contract: ModelContract) -> None:
            super().__init__()
            self.model = wrapped_model
            self.inner_wrapper = DinoV3IntermediateLayerWrapper(self.model, contract=wrapped_contract)

        def forward(self, pixel_values: Any) -> tuple[Any, ...]:
            return self.inner_wrapper.forward(pixel_values)

    return TorchDinoV3IntermediateLayerWrapper(model, contract)


def collect_onnx_op_types(onnx_path: Path) -> tuple[str, ...]:
    onnx = importlib.import_module("onnx")
    model = onnx.load(onnx_path)
    op_types: list[str] = []

    def visit_graph(graph: Any) -> None:
        for node in graph.node:
            op_types.append(str(node.op_type))
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    visit_graph(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for nested_graph in attribute.graphs:
                        visit_graph(nested_graph)

    visit_graph(model.graph)
    return tuple(op_types)


def assert_no_onnx_if_nodes(onnx_path: Path) -> None:
    op_types = collect_onnx_op_types(onnx_path)
    if "If" in op_types:
        raise ValueError(f"ONNX graph contains If node(s): {onnx_path}")
