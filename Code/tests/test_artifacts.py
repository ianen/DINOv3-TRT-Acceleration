import hashlib
from pathlib import Path

from dinov3_trt.artifacts import (
    ArtifactLayout,
    describe_artifact_file,
    find_weight_files,
    missing_required_assets,
    scan_assets,
    sha256_file,
)


def test_artifact_layout_uses_project_default_paths() -> None:
    layout = ArtifactLayout(Path("Artifacts"))

    assert layout.source_dir == Path("Artifacts/source/dinov3")
    assert layout.weights_dir == Path("Artifacts/weights/dinov3-vitl16-pretrain-lvd1689m")
    assert layout.onnx_path == Path("Artifacts/onnx/dinov3_vitl16_4out.onnx")
    assert layout.random_onnx_path == Path("Artifacts/onnx/dinov3_vitl16_4out.random.onnx")
    assert layout.engine_path("fp16") == Path("Artifacts/engines/dinov3_vitl16_4out.fp16.engine")
    assert layout.engine_path("bf16") == Path(
        "Artifacts/engines/dinov3_vitl16_4out.bf16.prefer.engine"
    )
    assert layout.random_engine_path("fp16") == Path(
        "Artifacts/engines/dinov3_vitl16_4out.random.fp16.engine"
    )
    assert layout.random_timing_cache_path == Path(
        "Artifacts/engines/dinov3_vitl16_4out.random.timing.cache"
    )
    assert layout.random_timing_cache_path_for("fp32") == Path(
        "Artifacts/engines/dinov3_vitl16_4out.random.fp32.timing.cache"
    )


def test_scan_assets_detects_source_weights_and_missing_engine(tmp_path: Path) -> None:
    layout = ArtifactLayout(tmp_path / "Artifacts")
    layout.create_directories()
    (layout.source_dir / "README.md").write_text("dinov3", encoding="utf-8")
    (layout.weights_dir / "model.safetensors").write_bytes(b"weights")
    layout.onnx_path.write_bytes(b"onnx")

    statuses = scan_assets(layout)

    assert statuses["source"].present is True
    assert statuses["weights"].present is True
    assert statuses["onnx"].present is True
    assert statuses["onnx-artifacts"].present is True
    assert statuses["random-onnx"].present is False
    assert statuses["fp16-engine"].present is False
    assert statuses["bf16-engine"].present is False
    assert missing_required_assets(layout, ("source", "weights", "fp16-engine")) == (
        statuses["fp16-engine"],
    )


def test_scan_assets_detects_random_poc_artifacts(tmp_path: Path) -> None:
    layout = ArtifactLayout(tmp_path / "Artifacts")
    layout.create_directories()
    layout.random_onnx_path.write_bytes(b"onnx")
    layout.random_engine_path("fp16").write_bytes(b"engine")
    layout.random_engine_path("fp32").write_bytes(b"engine")
    layout.random_timing_cache_path.write_bytes(b"cache")
    layout.random_timing_cache_path_for("fp32").write_bytes(b"cache")

    statuses = scan_assets(layout)

    assert statuses["random-onnx"].present is True
    assert statuses["random-fp16-engine"].present is True
    assert statuses["random-fp32-engine"].present is True
    assert statuses["random-timing-cache"].present is True
    assert statuses["random-fp32-timing-cache"].present is True
    assert statuses["engine-artifacts"].present is True


def test_scan_assets_lists_onnx_and_engine_artifact_files(tmp_path: Path) -> None:
    layout = ArtifactLayout(tmp_path / "Artifacts")
    layout.create_directories()
    layout.onnx_path.write_bytes(b"formal")
    layout.random_onnx_path.write_bytes(b"random")
    layout.engine_path("fp32").write_bytes(b"fp32")
    layout.engine_path("bf16").write_bytes(b"bf16")
    layout.random_timing_cache_path.write_bytes(b"cache")
    (layout.engines_dir / "ignored.lock").write_bytes(b"lock")

    statuses = scan_assets(layout)

    assert statuses["onnx-artifacts"].files == (
        layout.onnx_path,
        layout.random_onnx_path,
    )
    assert statuses["engine-artifacts"].files == (
        layout.engine_path("bf16"),
        layout.engine_path("fp32"),
        layout.random_timing_cache_path,
    )


def test_artifact_file_info_includes_size_and_optional_sha256(tmp_path: Path) -> None:
    payload = b"artifact"
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(payload)
    expected_digest = hashlib.sha256(payload).hexdigest()

    without_hash = describe_artifact_file(artifact)
    with_hash = describe_artifact_file(artifact, include_sha256=True)

    assert without_hash.size_bytes == len(payload)
    assert without_hash.sha256 is None
    assert with_hash.size_bytes == len(payload)
    assert with_hash.sha256 == expected_digest
    assert sha256_file(artifact, chunk_size=2) == expected_digest


def test_asset_status_file_info_uses_report_files(tmp_path: Path) -> None:
    layout = ArtifactLayout(tmp_path / "Artifacts")
    layout.create_directories()
    first_report = layout.reports_dir / "a.json"
    second_report = layout.reports_dir / "nested" / "b.md"
    first_report.write_text("{}", encoding="utf-8")
    second_report.parent.mkdir()
    second_report.write_text("report", encoding="utf-8")

    status = scan_assets(layout)["reports"]
    file_info = status.file_info()

    assert status.files == (first_report, second_report)
    assert [item.path for item in file_info] == [first_report, second_report]
    assert [item.size_bytes for item in file_info] == [2, 6]


def test_find_weight_files_returns_stable_safetensors_order(tmp_path: Path) -> None:
    (tmp_path / "z.safetensors").write_bytes(b"z")
    (tmp_path / "a.safetensors").write_bytes(b"a")
    (tmp_path / "model.pth").write_bytes(b"pth")
    (tmp_path / "ignore.bin").write_bytes(b"bin")

    assert [path.name for path in find_weight_files(tmp_path)] == [
        "a.safetensors",
        "model.pth",
        "z.safetensors",
    ]


def test_scan_assets_excludes_specified_files_from_report_scan(tmp_path: Path) -> None:
    layout = ArtifactLayout(tmp_path / "Artifacts")
    layout.create_directories()
    keeper = layout.reports_dir / "keep.json"
    self_manifest = layout.reports_dir / "artifact_manifest_formal_with_sha256.json"
    keeper.write_text("{}", encoding="utf-8")
    # Simulate the shell `>` redirect which creates the 0-byte target before the
    # Python process begins scanning the reports directory.
    self_manifest.write_bytes(b"")

    statuses = scan_assets(layout, exclude_files=[self_manifest])

    reports_status = statuses["reports"]
    assert reports_status.files == (keeper,)
    assert all(item.path != self_manifest for item in reports_status.file_info())


def test_scan_assets_excludes_files_from_onnx_and_engine_artifact_scans(
    tmp_path: Path,
) -> None:
    layout = ArtifactLayout(tmp_path / "Artifacts")
    layout.create_directories()
    layout.onnx_path.write_bytes(b"formal")
    extra_onnx = layout.onnx_path.parent / "scratch.onnx"
    extra_onnx.write_bytes(b"scratch")
    layout.engine_path("fp32").write_bytes(b"fp32")
    extra_engine = layout.engines_dir / "scratch.engine"
    extra_engine.write_bytes(b"scratch")

    statuses = scan_assets(layout, exclude_files=[extra_onnx, extra_engine])

    assert layout.onnx_path in statuses["onnx-artifacts"].files
    assert extra_onnx not in statuses["onnx-artifacts"].files
    assert layout.engine_path("fp32") in statuses["engine-artifacts"].files
    assert extra_engine not in statuses["engine-artifacts"].files


def test_missing_required_assets_honors_exclude_files(tmp_path: Path) -> None:
    layout = ArtifactLayout(tmp_path / "Artifacts")
    layout.create_directories()
    layout.onnx_path.write_bytes(b"onnx")

    # If we pretend the ONNX path is excluded, "onnx" should be reported missing.
    missing = missing_required_assets(layout, ("onnx",), exclude_files=[layout.onnx_path])
    assert [status.name for status in missing] == ["onnx"]

    # Without the exclusion it must remain present.
    assert missing_required_assets(layout, ("onnx",)) == ()
