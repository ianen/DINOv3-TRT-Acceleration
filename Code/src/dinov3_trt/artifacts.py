"""Artifact path and presence helpers for the DINOv3 TensorRT pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping


RequiredAsset = Literal[
    "source",
    "weights",
    "onnx",
    "fp32-engine",
    "fp16-engine",
    "bf16-engine",
    "reports",
]
EnginePrecision = Literal["fp32", "fp16", "bf16", "int8"]

DEFAULT_ARTIFACT_ROOT = Path("Artifacts")
DEFAULT_SOURCE_DIR = Path("source") / "dinov3"
DEFAULT_WEIGHTS_DIR = Path("weights") / "dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_ONNX_PATH = Path("onnx") / "dinov3_vitl16_4out.onnx"
DEFAULT_ENGINE_STEM = "dinov3_vitl16_4out"


@dataclass(frozen=True)
class ArtifactLayout:
    """Canonical artifact locations relative to an artifact root."""

    root: Path = DEFAULT_ARTIFACT_ROOT

    @property
    def source_dir(self) -> Path:
        return self.root / DEFAULT_SOURCE_DIR

    @property
    def weights_dir(self) -> Path:
        return self.root / DEFAULT_WEIGHTS_DIR

    @property
    def onnx_path(self) -> Path:
        return self.root / DEFAULT_ONNX_PATH

    @property
    def random_onnx_path(self) -> Path:
        return self.root / "onnx" / f"{DEFAULT_ENGINE_STEM}.random.onnx"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def engines_dir(self) -> Path:
        return self.root / "engines"

    @property
    def onnx_artifact_files(self) -> tuple[Path, ...]:
        return tuple(sorted(self.onnx_path.parent.glob("*.onnx")))

    @property
    def engine_artifact_files(self) -> tuple[Path, ...]:
        patterns = ("*.engine", "*.timing.cache")
        files = [
            path
            for pattern in patterns
            for path in self.engines_dir.glob(pattern)
            if path.is_file()
        ]
        return tuple(sorted(files))

    def engine_path(self, precision: EnginePrecision) -> Path:
        if precision == "bf16":
            return self.engines_dir / f"{DEFAULT_ENGINE_STEM}.bf16.prefer.engine"
        return self.engines_dir / f"{DEFAULT_ENGINE_STEM}.{precision}.engine"

    def random_engine_path(self, precision: EnginePrecision) -> Path:
        return self.engines_dir / f"{DEFAULT_ENGINE_STEM}.random.{precision}.engine"

    @property
    def random_timing_cache_path(self) -> Path:
        return self.engines_dir / f"{DEFAULT_ENGINE_STEM}.random.timing.cache"

    def random_timing_cache_path_for(self, precision: EnginePrecision) -> Path:
        if precision == "fp16":
            return self.random_timing_cache_path
        return self.engines_dir / f"{DEFAULT_ENGINE_STEM}.random.{precision}.timing.cache"

    def directories(self) -> tuple[Path, ...]:
        return (
            self.source_dir,
            self.weights_dir,
            self.onnx_path.parent,
            self.engines_dir,
            self.reports_dir,
        )

    def create_directories(self) -> None:
        for directory in self.directories():
            directory.mkdir(parents=True, exist_ok=True)

    def required_path(self, asset: RequiredAsset) -> Path:
        paths: Mapping[RequiredAsset, Path] = {
            "source": self.source_dir,
            "weights": self.weights_dir,
            "onnx": self.onnx_path,
            "fp32-engine": self.engine_path("fp32"),
            "fp16-engine": self.engine_path("fp16"),
            "bf16-engine": self.engine_path("bf16"),
            "reports": self.reports_dir,
        }
        return paths[asset]


@dataclass(frozen=True)
class ArtifactFileInfo:
    """Serializable file metadata for artifact manifests."""

    path: Path
    size_bytes: int
    sha256: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class AssetStatus:
    """Serializable presence result for one artifact category."""

    name: str
    path: Path
    present: bool
    detail: str
    files: tuple[Path, ...] = ()

    def file_info(self, *, include_sha256: bool = False) -> tuple[ArtifactFileInfo, ...]:
        paths = self.files if self.files else ((self.path,) if self.path.is_file() else ())
        return tuple(describe_artifact_file(path, include_sha256=include_sha256) for path in paths)

    def to_json(
        self,
        *,
        include_file_info: bool = False,
        include_sha256: bool = False,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "path": str(self.path),
            "present": self.present,
            "detail": self.detail,
            "files": [str(path) for path in self.files],
        }
        if include_file_info:
            payload["file_info"] = [
                file_info.to_json() for file_info in self.file_info(include_sha256=include_sha256)
            ]
        return payload


def _normalize_excluded(paths: Iterable[Path]) -> frozenset[Path]:
    """Return resolved absolute paths used as an exclusion set."""

    resolved: list[Path] = []
    for raw in paths:
        try:
            resolved.append(Path(raw).resolve())
        except OSError:
            resolved.append(Path(raw).absolute())
    return frozenset(resolved)


def _is_excluded(path: Path, excluded: frozenset[Path]) -> bool:
    if not excluded:
        return False
    try:
        candidate = path.resolve()
    except OSError:
        candidate = path.absolute()
    return candidate in excluded


def _has_any_file(path: Path, *, excluded: frozenset[Path] = frozenset()) -> bool:
    if not path.exists():
        return False
    return any(
        child.is_file() and not _is_excluded(child, excluded) for child in path.rglob("*")
    )


def _list_files(
    path: Path, *, excluded: frozenset[Path] = frozenset()
) -> tuple[Path, ...]:
    if not path.exists():
        return ()
    if path.is_file():
        if _is_excluded(path, excluded):
            return ()
        return (path,)
    return tuple(
        sorted(
            child
            for child in path.rglob("*")
            if child.is_file() and not _is_excluded(child, excluded)
        )
    )


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest for a file without loading it all into memory."""

    hasher = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def describe_artifact_file(path: Path, *, include_sha256: bool = False) -> ArtifactFileInfo:
    """Collect stable metadata for one artifact file."""

    digest = sha256_file(path) if include_sha256 else None
    return ArtifactFileInfo(path=path, size_bytes=path.stat().st_size, sha256=digest)


def find_weight_files(weights_dir: Path) -> tuple[Path, ...]:
    """Return supported DINOv3 weight files in stable order."""

    if not weights_dir.exists():
        return ()
    patterns = ("*.safetensors", "*.pth")
    files = [path for pattern in patterns for path in weights_dir.glob(pattern) if path.is_file()]
    return tuple(sorted(files))


def scan_assets(
    layout: ArtifactLayout,
    *,
    exclude_files: Iterable[Path] = (),
) -> dict[str, AssetStatus]:
    """Inspect expected project artifacts without mutating the filesystem.

    `exclude_files` lets callers omit specific files (typically the manifest
    being written out) from the recursive directory scans so the resulting
    manifest does not contain a self-referential 0-byte entry.
    """

    excluded = _normalize_excluded(exclude_files)

    source_present = _has_any_file(layout.source_dir, excluded=excluded)
    weight_files = tuple(
        path for path in find_weight_files(layout.weights_dir) if not _is_excluded(path, excluded)
    )
    onnx_present = layout.onnx_path.is_file() and not _is_excluded(layout.onnx_path, excluded)
    fp32_engine = layout.engine_path("fp32")
    fp16_engine = layout.engine_path("fp16")
    bf16_engine = layout.engine_path("bf16")
    random_fp32_engine = layout.random_engine_path("fp32")
    random_fp16_engine = layout.random_engine_path("fp16")
    random_fp32_timing_cache = layout.random_timing_cache_path_for("fp32")
    reports_present = layout.reports_dir.exists()
    report_files = _list_files(layout.reports_dir, excluded=excluded)
    onnx_artifact_files = tuple(
        path for path in layout.onnx_artifact_files if not _is_excluded(path, excluded)
    )
    engine_artifact_files = tuple(
        path for path in layout.engine_artifact_files if not _is_excluded(path, excluded)
    )

    return {
        "source": AssetStatus(
            name="source",
            path=layout.source_dir,
            present=source_present,
            detail="DINOv3 source tree detected" if source_present else "DINOv3 source tree missing",
        ),
        "weights": AssetStatus(
            name="weights",
            path=layout.weights_dir,
            present=bool(weight_files),
            detail=(
                f"{len(weight_files)} supported weight file(s) detected"
                if weight_files
                else "supported weights missing"
            ),
            files=weight_files,
        ),
        "onnx": AssetStatus(
            name="onnx",
            path=layout.onnx_path,
            present=onnx_present,
            detail="4-output ONNX detected" if onnx_present else "4-output ONNX missing",
        ),
        "random-onnx": AssetStatus(
            name="random-onnx",
            path=layout.random_onnx_path,
            present=layout.random_onnx_path.is_file(),
            detail=(
                "random-weight 4-output ONNX detected"
                if layout.random_onnx_path.is_file()
                else "random-weight 4-output ONNX missing"
            ),
        ),
        "onnx-artifacts": AssetStatus(
            name="onnx-artifacts",
            path=layout.onnx_path.parent,
            present=bool(onnx_artifact_files),
            detail=(
                f"{len(onnx_artifact_files)} ONNX artifact file(s) detected"
                if onnx_artifact_files
                else "ONNX artifact files missing"
            ),
            files=onnx_artifact_files,
        ),
        "fp32-engine": AssetStatus(
            name="fp32-engine",
            path=fp32_engine,
            present=fp32_engine.is_file(),
            detail="FP32 TensorRT engine detected" if fp32_engine.is_file() else "FP32 engine missing",
        ),
        "fp16-engine": AssetStatus(
            name="fp16-engine",
            path=fp16_engine,
            present=fp16_engine.is_file(),
            detail="FP16 TensorRT engine detected" if fp16_engine.is_file() else "FP16 engine missing",
        ),
        "bf16-engine": AssetStatus(
            name="bf16-engine",
            path=bf16_engine,
            present=bf16_engine.is_file(),
            detail="BF16 TensorRT engine detected" if bf16_engine.is_file() else "BF16 engine missing",
        ),
        "random-fp16-engine": AssetStatus(
            name="random-fp16-engine",
            path=random_fp16_engine,
            present=random_fp16_engine.is_file(),
            detail=(
                "random-weight FP16 TensorRT engine detected"
                if random_fp16_engine.is_file()
                else "random-weight FP16 engine missing"
            ),
        ),
        "random-fp32-engine": AssetStatus(
            name="random-fp32-engine",
            path=random_fp32_engine,
            present=random_fp32_engine.is_file(),
            detail=(
                "random-weight FP32 TensorRT engine detected"
                if random_fp32_engine.is_file()
                else "random-weight FP32 engine missing"
            ),
        ),
        "random-timing-cache": AssetStatus(
            name="random-timing-cache",
            path=layout.random_timing_cache_path,
            present=layout.random_timing_cache_path.is_file(),
            detail=(
                "random-weight TensorRT timing cache detected"
                if layout.random_timing_cache_path.is_file()
                else "random-weight TensorRT timing cache missing"
            ),
        ),
        "random-fp32-timing-cache": AssetStatus(
            name="random-fp32-timing-cache",
            path=random_fp32_timing_cache,
            present=random_fp32_timing_cache.is_file(),
            detail=(
                "random-weight FP32 TensorRT timing cache detected"
                if random_fp32_timing_cache.is_file()
                else "random-weight FP32 TensorRT timing cache missing"
            ),
        ),
        "engine-artifacts": AssetStatus(
            name="engine-artifacts",
            path=layout.engines_dir,
            present=bool(engine_artifact_files),
            detail=(
                f"{len(engine_artifact_files)} TensorRT engine/cache artifact file(s) detected"
                if engine_artifact_files
                else "TensorRT engine/cache artifact files missing"
            ),
            files=engine_artifact_files,
        ),
        "reports": AssetStatus(
            name="reports",
            path=layout.reports_dir,
            present=reports_present,
            detail="reports directory detected" if reports_present else "reports directory missing",
            files=report_files,
        ),
    }


def missing_required_assets(
    layout: ArtifactLayout,
    required: tuple[RequiredAsset, ...],
    *,
    exclude_files: Iterable[Path] = (),
) -> tuple[AssetStatus, ...]:
    statuses = scan_assets(layout, exclude_files=exclude_files)
    return tuple(statuses[name] for name in required if not statuses[name].present)
