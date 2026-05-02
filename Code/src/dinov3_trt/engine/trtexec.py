"""Build `trtexec` command lines from the project contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from dinov3_trt.contracts import DINO_VITL16_224_CONTRACT, ModelContract

Precision = Literal["fp32", "fp16", "bf16", "int8", "fp8"]
PrecisionConstraint = Literal["obey", "prefer"]


@dataclass(frozen=True)
class ShapeProfile:
    """Dynamic batch / static resolution TensorRT profile."""

    input_name: str = "pixel_values"
    min_batch: int = 1
    opt_batch: int = 8
    max_batch: int = 32
    channels: int = 3
    height: int = 224
    width: int = 224

    def _shape(self, batch_size: int) -> str:
        if batch_size < 1:
            raise ValueError("batch size must be >= 1")
        return f"{self.input_name}:{batch_size}x{self.channels}x{self.height}x{self.width}"

    @property
    def min_shape(self) -> str:
        return self._shape(self.min_batch)

    @property
    def opt_shape(self) -> str:
        return self._shape(self.opt_batch)

    @property
    def max_shape(self) -> str:
        return self._shape(self.max_batch)


@dataclass(frozen=True)
class TrtExecConfig:
    """Configuration for one TensorRT engine build.

    V1.0.2 additions (ADR-013): ``additional_profiles``, ``builder_optimization_level``,
    ``persistent_cache_size_mb``, ``enable_sparsity``. All default to V1.0.1
    behavior so existing call sites are unaffected.
    """

    onnx_path: Path
    engine_path: Path
    precision: Precision
    profile: ShapeProfile = ShapeProfile()
    workspace_gb: int = 4
    timing_cache_path: Path | None = None
    skip_inference: bool = True
    no_tf32: bool = True
    profiling_verbosity: Literal["layer_names_only", "detailed", "none"] = "detailed"
    precision_constraints: PrecisionConstraint | None = None
    layer_precisions: tuple[str, ...] = ()
    layer_output_types: tuple[str, ...] = ()
    contract: ModelContract = DINO_VITL16_224_CONTRACT
    # V1.0.2 ADR-013: additional optimization profiles appended after `profile`.
    # Each entry produces an extra ``--profile --minShapes ... --optShapes ... --maxShapes ...``
    # block in the trtexec invocation. trtexec selects per-batch-size tactics
    # independently per profile, often gaining 5–10% on extreme batch ranges.
    additional_profiles: tuple[ShapeProfile, ...] = ()
    # V1.0.2 ADR-013: trtexec ``--builderOptimizationLevel`` (0–5). None keeps
    # trtexec default (3 in TRT 10.x). Level 5 makes the first build ~3× slower
    # but is amortised by ``timing_cache_path`` on subsequent builds, and
    # typically yields +2–5% engine quality.
    builder_optimization_level: int | None = None
    # V1.0.2 ADR-013: trtexec ``--persistentCacheSize`` in MB. Configures the
    # CUDA L2 persistent cache for kernel-private weight caching. None keeps
    # the TRT default. Useful for ViT attention key/value reuse patterns.
    persistent_cache_size_mb: int | None = None
    # V1.0.2 ADR-016: trtexec ``--sparsity=enable`` for 2:4 structured sparsity
    # tactic selection. The ONNX itself must already carry sparse weights;
    # this flag merely tells trtexec to pick sparse Tensor Core kernels.
    enable_sparsity: bool = False

    def __post_init__(self) -> None:
        if self.workspace_gb < 1:
            raise ValueError("workspace_gb must be >= 1")
        if self.profile.height != self.contract.image_size or self.profile.width != self.contract.image_size:
            raise ValueError("profile spatial dimensions must match the static model contract")
        if self.precision not in ("fp16", "bf16", "int8") and (
            self.precision_constraints is not None
            or self.layer_precisions
            or self.layer_output_types
        ):
            raise ValueError(
                "precision constraints are only supported for FP16/BF16/INT8 mixed-precision engines"
            )
        for extra in self.additional_profiles:
            if (
                extra.height != self.contract.image_size
                or extra.width != self.contract.image_size
            ):
                raise ValueError(
                    "additional profile spatial dimensions must match the static model contract"
                )
        if self.builder_optimization_level is not None and not (
            0 <= self.builder_optimization_level <= 5
        ):
            raise ValueError("builder_optimization_level must be in [0, 5]")
        if self.persistent_cache_size_mb is not None and self.persistent_cache_size_mb < 0:
            raise ValueError("persistent_cache_size_mb must be non-negative")


def _path(value: Path) -> str:
    return str(value)


def build_trtexec_command(config: TrtExecConfig) -> list[str]:
    """Return a `trtexec` argv list for the project's static-resolution profile.

    Supports V1.0.2 multi-profile builds (ADR-013): when
    ``config.additional_profiles`` is non-empty, each extra profile is appended
    after a ``--profile`` separator so trtexec selects per-profile tactics.
    """

    command = [
        "trtexec",
        f"--onnx={_path(config.onnx_path)}",
        f"--saveEngine={_path(config.engine_path)}",
        f"--minShapes={config.profile.min_shape}",
        f"--optShapes={config.profile.opt_shape}",
        f"--maxShapes={config.profile.max_shape}",
        f"--memPoolSize=workspace:{config.workspace_gb}G",
        f"--profilingVerbosity={config.profiling_verbosity}",
        "--verbose",
    ]
    for extra in config.additional_profiles:
        command.append("--profile")
        command.extend(
            [
                f"--minShapes={extra.min_shape}",
                f"--optShapes={extra.opt_shape}",
                f"--maxShapes={extra.max_shape}",
            ]
        )
    if config.no_tf32:
        command.append("--noTF32")
    if config.precision in ("fp16", "bf16", "int8", "fp8"):
        command.append(f"--{config.precision}")
    if config.precision in ("fp16", "bf16", "int8"):
        precision_constraints = config.precision_constraints
        if precision_constraints is None and (
            config.layer_precisions or config.layer_output_types
        ):
            precision_constraints = "prefer"
        if precision_constraints is not None:
            command.append(f"--precisionConstraints={precision_constraints}")
    if config.layer_precisions:
        command.append(f"--layerPrecisions={','.join(config.layer_precisions)}")
    if config.layer_output_types:
        command.append(f"--layerOutputTypes={','.join(config.layer_output_types)}")
    if config.timing_cache_path is not None:
        command.append(f"--timingCacheFile={_path(config.timing_cache_path)}")
    if config.builder_optimization_level is not None:
        command.append(
            f"--builderOptimizationLevel={config.builder_optimization_level}"
        )
    if config.persistent_cache_size_mb is not None:
        command.append(f"--persistentCacheSize={config.persistent_cache_size_mb}")
    if config.enable_sparsity:
        command.append("--sparsity=enable")
    if config.skip_inference:
        command.append("--skipInference")
    return command


def quote_for_display(command: Sequence[str]) -> str:
    """Return a shell-readable command string without choosing a shell."""

    return " ".join(f'"{part}"' if " " in part else part for part in command)
