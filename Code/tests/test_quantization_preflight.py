import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from dinov3_trt.infer.image_eval import write_image_manifest
from dinov3_trt.quantization.preflight import (
    CudaStatus,
    DependencyRequirement,
    DependencyStatus,
    ManifestStatus,
    QuantizationPreflightReport,
    _resolve_attribute,
    _version_for,
    build_preflight_report,
    check_cuda,
    check_dependency,
    check_manifest,
)


def test_check_dependency_reports_missing_module() -> None:
    status = check_dependency(
        DependencyRequirement(
            name="missing",
            module="dinov3_trt_missing_dependency",
            distribution="dinov3-trt-missing-dependency",
        )
    )

    assert not status.installed
    assert status.version is None
    assert "not importable" in status.detail


def test_check_manifest_reads_image_count(tmp_path: Path) -> None:
    image = tmp_path / "class-a" / "image.jpg"
    image.parent.mkdir()
    image.write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.json"
    write_image_manifest(
        manifest,
        image_root=tmp_path,
        images=(image,),
        seed=1,
        split="calib",
    )

    status = check_manifest("calibration", manifest)

    assert status.present
    assert status.image_count == 1


def test_preflight_report_ready_requires_deps_cuda_and_data(tmp_path: Path) -> None:
    calib_image = tmp_path / "calib.jpg"
    eval_image = tmp_path / "eval.jpg"
    calib_image.write_bytes(b"placeholder")
    eval_image.write_bytes(b"placeholder")
    calib_manifest = tmp_path / "calib.json"
    eval_manifest = tmp_path / "eval.json"
    write_image_manifest(
        calib_manifest,
        image_root=tmp_path,
        images=(calib_image,),
        seed=1,
        split="calib",
    )
    write_image_manifest(
        eval_manifest,
        image_root=tmp_path,
        images=(eval_image,),
        seed=1,
        split="eval",
    )
    dependencies = (
        DependencyStatus(
            name="ModelOpt",
            module="modelopt",
            distribution="nvidia-modelopt",
            installed=True,
            version="0.43.0",
            detail="ok",
        ),
    )
    cuda = CudaStatus(
        torch_installed=True,
        available=True,
        device_count=1,
        device_name="RTX 5080",
        detail="ok",
    )

    report = build_preflight_report(
        calib_manifest=calib_manifest,
        eval_manifest=eval_manifest,
        dependencies=dependencies,
        cuda=cuda,
    )

    assert report.ready
    assert report.dependencies_ready
    assert report.data_ready
    assert report.to_json()["cuda_ready"]


def test_preflight_report_not_ready_when_manifest_missing(tmp_path: Path) -> None:
    dependencies = (
        DependencyStatus(
            name="ModelOpt",
            module="modelopt",
            distribution="nvidia-modelopt",
            installed=True,
            version="0.43.0",
            detail="ok",
        ),
    )
    cuda = CudaStatus(
        torch_installed=True,
        available=True,
        device_count=1,
        device_name="RTX 5080",
        detail="ok",
    )

    report = build_preflight_report(
        calib_manifest=tmp_path / "missing-calib.json",
        eval_manifest=tmp_path / "missing-eval.json",
        dependencies=dependencies,
        cuda=cuda,
    )

    assert not report.ready
    assert report.dependencies_ready
    assert not report.data_ready


# --- additional coverage: helpers + check_cuda + check_dependency edges ---


def test_resolve_attribute_traverses_dotted_path() -> None:
    module = SimpleNamespace(sub=SimpleNamespace(attr="value"))
    assert _resolve_attribute(module, "sub.attr") == "value"  # type: ignore[arg-type]


def test_resolve_attribute_raises_on_missing() -> None:
    module = SimpleNamespace()
    with pytest.raises(AttributeError):
        _resolve_attribute(module, "missing")  # type: ignore[arg-type]


def test_version_for_returns_none_when_distribution_absent() -> None:
    assert _version_for("dinov3-trt-no-such-distribution-xyz") is None


def test_check_dependency_with_required_attribute_present(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = ModuleType("fake_dinov3_trt_dep")
    fake_module.__version__ = "1.2.3"  # type: ignore[attr-defined]
    setattr(fake_module, "MARKER", object())
    monkeypatch.setitem(sys.modules, "fake_dinov3_trt_dep", fake_module)

    requirement = DependencyRequirement(
        name="FakeDep",
        module="fake_dinov3_trt_dep",
        distribution="fake-dinov3-trt-dep",
        required_attribute="MARKER",
    )
    with patch("importlib.util.find_spec", return_value=object()):
        status = check_dependency(requirement)

    assert status.installed
    assert "required attribute present" in status.detail


def test_check_dependency_with_required_attribute_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = ModuleType("fake_dinov3_trt_dep_missing_attr")
    monkeypatch.setitem(sys.modules, "fake_dinov3_trt_dep_missing_attr", fake_module)

    requirement = DependencyRequirement(
        name="FakeDep",
        module="fake_dinov3_trt_dep_missing_attr",
        distribution="fake-dinov3-trt-dep",
        required_attribute="MISSING_MARKER",
    )
    with patch("importlib.util.find_spec", return_value=object()):
        status = check_dependency(requirement)

    assert not status.installed
    assert "required attribute missing" in status.detail


def test_check_cuda_when_torch_not_installed() -> None:
    with patch("importlib.util.find_spec", return_value=None):
        status = check_cuda()

    assert not status.torch_installed
    assert not status.available
    assert "torch module not importable" in status.detail


def test_check_cuda_when_probe_fails() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: (_ for _ in ()).throw(RuntimeError("driver fail"))
        ),
    )
    with patch("importlib.util.find_spec", return_value=object()), patch(
        "importlib.import_module", return_value=fake_torch
    ):
        status = check_cuda()

    assert status.torch_installed
    assert not status.available
    assert "torch CUDA probe failed" in status.detail


def test_check_cuda_when_available_returns_device_info() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda idx: "RTX 5080",
        ),
    )
    with patch("importlib.util.find_spec", return_value=object()), patch(
        "importlib.import_module", return_value=fake_torch
    ):
        status = check_cuda()

    assert status.torch_installed
    assert status.available
    assert status.device_count == 1
    assert status.device_name == "RTX 5080"


def test_check_cuda_when_unavailable() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            get_device_name=lambda idx: "n/a",
        ),
    )
    with patch("importlib.util.find_spec", return_value=object()), patch(
        "importlib.import_module", return_value=fake_torch
    ):
        status = check_cuda()

    assert status.torch_installed
    assert not status.available


def test_check_manifest_missing_path_returns_absent_status(tmp_path: Path) -> None:
    status = check_manifest("calibration", tmp_path / "does-not-exist.json")

    assert not status.present


def test_manifest_status_image_count_field() -> None:
    status = ManifestStatus(
        name="x", path=Path("/tmp/x.json"), present=True, image_count=42, detail="ok"
    )
    assert status.image_count == 42
    payload = status.to_json()
    assert payload["name"] == "x"
    assert payload["image_count"] == 42


def test_quantization_preflight_report_to_json_roundtrip() -> None:
    """to_json round-trip exercises the dataclass fields."""

    deps = (
        DependencyStatus(
            name="X", module="x", distribution="x", installed=True, version="1.0", detail="ok"
        ),
    )
    cuda = CudaStatus(
        torch_installed=True, available=True, device_count=1, device_name="X", detail="ok"
    )
    manifest = ManifestStatus(
        name="calib", path=Path("/tmp/c"), present=True, image_count=1, detail="ok"
    )

    report = QuantizationPreflightReport(
        dependencies=deps,
        cuda=cuda,
        manifests=(manifest,),
    )
    assert report.dependencies_ready
    payload = report.to_json()
    assert "dependencies" in payload
    assert "cuda" in payload
    assert "manifests" in payload
