"""Preflight checks for the ModelOpt INT8 quantization path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import importlib.metadata
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from dinov3_trt.infer.image_eval import read_image_manifest


@dataclass(frozen=True)
class DependencyRequirement:
    """A Python package/module required by the INT8 path."""

    name: str
    module: str
    distribution: str
    required_attribute: str | None = None


@dataclass(frozen=True)
class DependencyStatus:
    """Serializable status for one dependency requirement."""

    name: str
    module: str
    distribution: str
    installed: bool
    version: str | None
    detail: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CudaStatus:
    """Serializable torch CUDA runtime status."""

    torch_installed: bool
    available: bool
    device_count: int | None
    device_name: str | None
    detail: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ManifestStatus:
    """Serializable status for one image manifest."""

    name: str
    path: Path
    present: bool
    image_count: int
    detail: str

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": str(self.path),
            "present": self.present,
            "image_count": self.image_count,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class QuantizationPreflightReport:
    """Combined readiness report for P4 ModelOpt quantization."""

    dependencies: tuple[DependencyStatus, ...]
    cuda: CudaStatus
    manifests: tuple[ManifestStatus, ...]

    @property
    def dependencies_ready(self) -> bool:
        return all(dependency.installed for dependency in self.dependencies)

    @property
    def data_ready(self) -> bool:
        return all(manifest.present for manifest in self.manifests)

    @property
    def ready(self) -> bool:
        return self.dependencies_ready and self.cuda.available and self.data_ready

    def to_json(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "dependencies_ready": self.dependencies_ready,
            "cuda_ready": self.cuda.available,
            "data_ready": self.data_ready,
            "dependencies": [dependency.to_json() for dependency in self.dependencies],
            "cuda": self.cuda.to_json(),
            "manifests": [manifest.to_json() for manifest in self.manifests],
        }


MODEL_OPT_INT8_REQUIREMENT = DependencyRequirement(
    name="NVIDIA ModelOpt INT8 config",
    module="modelopt.torch.quantization",
    distribution="nvidia-modelopt",
    required_attribute="INT8_DEFAULT_CFG",
)

DEFAULT_DEPENDENCIES: tuple[DependencyRequirement, ...] = (
    MODEL_OPT_INT8_REQUIREMENT,
    DependencyRequirement(
        name="NVIDIA ModelOpt ONNX PTQ",
        module="modelopt.onnx.quantization",
        distribution="nvidia-modelopt",
        required_attribute="quantize",
    ),
    DependencyRequirement(name="Polygraphy", module="polygraphy", distribution="polygraphy"),
    DependencyRequirement(name="ONNX Runtime", module="onnxruntime", distribution="onnxruntime"),
    DependencyRequirement(name="PyTorch", module="torch", distribution="torch"),
    DependencyRequirement(name="TensorRT", module="tensorrt", distribution="tensorrt"),
)


def _version_for(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _resolve_attribute(module: ModuleType, attribute_path: str) -> object:
    value: object = module
    for part in attribute_path.split("."):
        value = getattr(value, part)
    return value


def check_dependency(requirement: DependencyRequirement) -> DependencyStatus:
    """Return import/version status for one Python dependency."""

    version = _version_for(requirement.distribution)
    try:
        spec = importlib.util.find_spec(requirement.module)
    except Exception as exc:
        return DependencyStatus(
            name=requirement.name,
            module=requirement.module,
            distribution=requirement.distribution,
            installed=False,
            version=version,
            detail=f"module probe failed: {exc}",
        )
    if spec is None:
        return DependencyStatus(
            name=requirement.name,
            module=requirement.module,
            distribution=requirement.distribution,
            installed=False,
            version=version,
            detail=f"module not importable: {requirement.module}",
        )
    if requirement.required_attribute is None:
        return DependencyStatus(
            name=requirement.name,
            module=requirement.module,
            distribution=requirement.distribution,
            installed=True,
            version=version,
            detail="module importable",
        )
    try:
        module = importlib.import_module(requirement.module)
        _resolve_attribute(module, requirement.required_attribute)
    except Exception as exc:
        return DependencyStatus(
            name=requirement.name,
            module=requirement.module,
            distribution=requirement.distribution,
            installed=False,
            version=version,
            detail=f"required attribute missing: {requirement.required_attribute} ({exc})",
        )
    return DependencyStatus(
        name=requirement.name,
        module=requirement.module,
        distribution=requirement.distribution,
        installed=True,
        version=version,
        detail=f"required attribute present: {requirement.required_attribute}",
    )


def check_cuda() -> CudaStatus:
    """Return torch CUDA readiness without requiring torch at import time."""

    if importlib.util.find_spec("torch") is None:
        return CudaStatus(
            torch_installed=False,
            available=False,
            device_count=None,
            device_name=None,
            detail="torch module not importable",
        )
    try:
        torch: Any = importlib.import_module("torch")
        available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if available else 0
        device_name = str(torch.cuda.get_device_name(0)) if available and device_count else None
    except Exception as exc:
        return CudaStatus(
            torch_installed=True,
            available=False,
            device_count=None,
            device_name=None,
            detail=f"torch CUDA probe failed: {exc}",
        )
    return CudaStatus(
        torch_installed=True,
        available=available,
        device_count=device_count,
        device_name=device_name,
        detail="torch CUDA available" if available else "torch CUDA unavailable",
    )


def check_manifest(name: str, path: Path) -> ManifestStatus:
    """Return existence/count status for an image manifest."""

    if not path.is_file():
        return ManifestStatus(
            name=name,
            path=path,
            present=False,
            image_count=0,
            detail="manifest file missing",
        )
    try:
        image_paths = read_image_manifest(path)
    except Exception as exc:
        return ManifestStatus(
            name=name,
            path=path,
            present=False,
            image_count=0,
            detail=f"manifest unreadable: {exc}",
        )
    return ManifestStatus(
        name=name,
        path=path,
        present=True,
        image_count=len(image_paths),
        detail="manifest readable",
    )


def build_preflight_report(
    *,
    calib_manifest: Path,
    eval_manifest: Path,
    dependencies: tuple[DependencyStatus, ...] | None = None,
    cuda: CudaStatus | None = None,
    requirements: tuple[DependencyRequirement, ...] = DEFAULT_DEPENDENCIES,
) -> QuantizationPreflightReport:
    """Build a full ModelOpt INT8 preflight report."""

    dependency_statuses = (
        dependencies
        if dependencies is not None
        else tuple(check_dependency(requirement) for requirement in requirements)
    )
    cuda_status = cuda if cuda is not None else check_cuda()
    return QuantizationPreflightReport(
        dependencies=dependency_statuses,
        cuda=cuda_status,
        manifests=(
            check_manifest("calibration", calib_manifest),
            check_manifest("evaluation", eval_manifest),
        ),
    )
