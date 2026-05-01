import csv
import json
from pathlib import Path

import pytest

from dinov3_trt.reports.benchmark_figures import (
    DEFAULT_BENCHMARK_FIGURE_SPECS,
    DEFAULT_COSINE_FIGURE_SPECS,
    DEFAULT_LAYER_ABLATION_FIGURE_SPECS,
    DEFAULT_TRADEOFF_FIGURE_SPECS,
    BenchmarkFigureRow,
    BenchmarkFigureSpec,
    CosineEvalReport,
    CosineFigureSpec,
    LayerAblationFigurePoint,
    LayerAblationFigureSpec,
    TradeoffFigureSpec,
    TradeoffPoint,
    build_benchmark_figures,
    build_cosine_figures,
    build_layer_ablation_figures,
    build_tradeoff_figures,
    extract_cosine_bars,
    extract_layer_ablation_points,
    extract_tradeoff_points,
    load_benchmark_matrix_csv,
    render_cosine_svg,
    render_layer_ablation_svg,
    render_speedup_svg,
    render_tradeoff_svg,
)


def test_default_figure_specs_cover_bf16_int8_and_cpp() -> None:
    names = {spec.name for spec in DEFAULT_BENCHMARK_FIGURE_SPECS}

    assert "trtexec-bf16-speedup" in names
    assert "trtexec-int8-speedup" in names
    assert "cpp-runtime-speedup" in names


def test_load_benchmark_matrix_csv_extracts_plotted_fields(tmp_path: Path) -> None:
    matrix_csv = tmp_path / "matrix.csv"
    with matrix_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("runtime", "candidate", "reference", "batch_size", "latency_speedup"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "runtime": "trtexec",
                "candidate": "BF16-prefer",
                "reference": "FP32",
                "batch_size": "8",
                "latency_speedup": "2.81",
            }
        )

    rows = load_benchmark_matrix_csv(matrix_csv)

    assert rows == [
                BenchmarkFigureRow(
                    runtime="trtexec",
                    candidate="BF16-prefer",
                    reference="FP32",
                    batch_size=8,
                    resolution=224,
                    latency_speedup=2.81,
                )
            ]


def test_render_speedup_svg_contains_bars_and_labels() -> None:
    svg = render_speedup_svg(
        BenchmarkFigureSpec(
            name="bf16",
            title="BF16 speedup",
            runtime="trtexec",
            reference="FP32",
            candidates=("BF16-prefer",),
            output_filename="bf16.svg",
        ),
        (
            BenchmarkFigureRow(
                runtime="trtexec",
                candidate="BF16-prefer",
                reference="FP32",
                batch_size=1,
                resolution=224,
                latency_speedup=2.45,
            ),
        ),
    )

    assert svg.startswith("<svg")
    assert "BF16 speedup" in svg
    assert "<rect" in svg
    assert "2.45x" in svg


def test_render_speedup_svg_disambiguates_mixed_resolutions() -> None:
    svg = render_speedup_svg(
        BenchmarkFigureSpec(
            name="bf16",
            title="BF16 speedup",
            runtime="trtexec",
            reference="FP32",
            candidates=("BF16-prefer",),
            output_filename="bf16.svg",
        ),
        (
            BenchmarkFigureRow(
                runtime="trtexec",
                candidate="BF16-prefer",
                reference="FP32",
                batch_size=1,
                resolution=224,
                latency_speedup=2.45,
            ),
            BenchmarkFigureRow(
                runtime="trtexec",
                candidate="BF16-prefer",
                reference="FP32",
                batch_size=1,
                resolution=336,
                latency_speedup=2.80,
            ),
        ),
    )

    assert "R224 B1" in svg
    assert "R336 B1" in svg


def test_build_benchmark_figures_writes_manifest_and_svg(tmp_path: Path) -> None:
    matrix_csv = tmp_path / "matrix.csv"
    with matrix_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("runtime", "candidate", "reference", "batch_size", "latency_speedup"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "runtime": "trtexec",
                "candidate": "BF16-prefer",
                "reference": "FP32",
                "batch_size": "1",
                "latency_speedup": "2.45",
            }
        )

    output_dir = tmp_path / "figures"
    manifest = build_benchmark_figures(
        matrix_csv,
        output_dir,
        specs=(
            BenchmarkFigureSpec(
                name="bf16",
                title="BF16 speedup",
                runtime="trtexec",
                reference="FP32",
                candidates=("BF16-prefer",),
                output_filename="bf16.svg",
            ),
        ),
    )

    assert (output_dir / "bf16.svg").exists()
    manifest_path = output_dir / "benchmark_figures_manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["figures"][0]["name"] == "bf16"
    assert manifest["figures"][0]["row_count"] == 1


def _write_eval_report(path: Path, *, image_size: int, cosine_min: tuple[float, float, float, float], cosine_mean: tuple[float, float, float, float]) -> None:
    output_names = ("feat_layer_4", "feat_layer_12", "feat_layer_16", "feat_layer_20")
    payload = {
        "reference_engine": "fp32.engine",
        "candidate_engine": "bf16.prefer.engine",
        "image_count": 1000,
        "batch_size": 8,
        "image_size": image_size,
        "outputs": [
            {
                "name": name,
                "cosine_similarity_min": cmin,
                "cosine_similarity_mean": cmean,
                "max_abs_error": 1.0,
                "root_mean_square_error": 0.01,
            }
            for name, cmin, cmean in zip(output_names, cosine_min, cosine_mean)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_default_cosine_figure_specs_cover_three_resolutions_and_metrics() -> None:
    metrics = {spec.metric for spec in DEFAULT_COSINE_FIGURE_SPECS}
    resolutions: set[int] = set()
    for spec in DEFAULT_COSINE_FIGURE_SPECS:
        resolutions.update(report.resolution for report in spec.reports)

    assert metrics == {"cosine_min", "cosine_mean"}
    assert resolutions == {224, 336, 518}


def test_extract_cosine_bars_pulls_per_output_min_values(tmp_path: Path) -> None:
    _write_eval_report(
        tmp_path / "eval_imagenette1000_fp32_vs_bf16_prefer.json",
        image_size=224,
        cosine_min=(0.999933, 0.999664, 0.998943, 0.998749),
        cosine_mean=(0.999953, 0.999788, 0.999377, 0.999127),
    )
    _write_eval_report(
        tmp_path / "eval_imagenette1000_r336_fp32_vs_bf16_prefer.json",
        image_size=336,
        cosine_min=(0.999891, 0.999276, 0.998394, 0.998493),
        cosine_mean=(0.999947, 0.999766, 0.999432, 0.99936),
    )
    _write_eval_report(
        tmp_path / "eval_imagenette1000_r518_fp32_vs_bf16_prefer.json",
        image_size=518,
        cosine_min=(0.999868, 0.999075, 0.998604, 0.999171),
        cosine_mean=(0.999945, 0.999800, 0.999655, 0.999721),
    )

    spec = DEFAULT_COSINE_FIGURE_SPECS[0]
    bars, missing = extract_cosine_bars(spec, tmp_path)

    assert missing == []
    assert len(bars) == 12  # 3 resolutions x 4 outputs
    by_key = {(bar.output_name, bar.resolution): bar.value for bar in bars}
    assert by_key[("feat_layer_20", 224)] == pytest.approx(0.998749)
    assert by_key[("feat_layer_20", 518)] == pytest.approx(0.999171)
    # cosine_min for r336 feat_layer_20 was 0.998493 in the canonical multi-resolution table
    assert by_key[("feat_layer_20", 336)] == pytest.approx(0.998493)


def test_extract_cosine_bars_raises_when_report_missing(tmp_path: Path) -> None:
    spec = CosineFigureSpec(
        name="bf16-cosine",
        title="cosine",
        candidate="BF16-prefer",
        reference="FP32",
        reports=(
            CosineEvalReport(
                label="R224",
                filename="missing_eval.json",
                resolution=224,
            ),
        ),
        output_filename="cosine.svg",
    )

    with pytest.raises(FileNotFoundError, match="missing_eval.json"):
        extract_cosine_bars(spec, tmp_path)


def test_extract_cosine_bars_supports_allow_missing(tmp_path: Path) -> None:
    spec = CosineFigureSpec(
        name="bf16-cosine",
        title="cosine",
        candidate="BF16-prefer",
        reference="FP32",
        reports=(
            CosineEvalReport(
                label="R224",
                filename="missing_eval.json",
                resolution=224,
            ),
        ),
        output_filename="cosine.svg",
    )

    bars, missing = extract_cosine_bars(spec, tmp_path, allow_missing=True)

    assert bars == []
    assert missing == ["missing_eval.json"]


def test_render_cosine_svg_zooms_axis_to_show_high_cosine_bars(tmp_path: Path) -> None:
    spec = CosineFigureSpec(
        name="bf16-cosine-min",
        title="BF16 cosine_min",
        candidate="BF16-prefer",
        reference="FP32",
        reports=(
            CosineEvalReport(
                label="R224",
                filename="eval_imagenette1000_fp32_vs_bf16_prefer.json",
                resolution=224,
            ),
        ),
        output_filename="cosine.svg",
        metric="cosine_min",
    )
    _write_eval_report(
        tmp_path / "eval_imagenette1000_fp32_vs_bf16_prefer.json",
        image_size=224,
        cosine_min=(0.999933, 0.999664, 0.998943, 0.998749),
        cosine_mean=(0.999953, 0.999788, 0.999377, 0.999127),
    )

    bars, _missing = extract_cosine_bars(spec, tmp_path)
    svg = render_cosine_svg(spec, bars)

    assert svg.startswith("<svg")
    assert "BF16 cosine_min" in svg
    assert "feat_layer_4" in svg
    assert "feat_layer_20" in svg
    # The lowest cosine_min in this fixture is 0.998749, so the y axis floor must
    # be 0.997 or below to leave headroom; "1.0000" appears as the top tick.
    assert "1.0000" in svg
    # Ensure a tick label between 0.997 and 0.999 shows up (zoomed cosine axis).
    assert "0.998" in svg or "0.997" in svg


def test_build_cosine_figures_writes_svg_and_manifest(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    output_dir = tmp_path / "figures"
    _write_eval_report(
        reports_dir / "eval_imagenette1000_fp32_vs_bf16_prefer.json",
        image_size=224,
        cosine_min=(0.999933, 0.999664, 0.998943, 0.998749),
        cosine_mean=(0.999953, 0.999788, 0.999377, 0.999127),
    )
    _write_eval_report(
        reports_dir / "eval_imagenette1000_r336_fp32_vs_bf16_prefer.json",
        image_size=336,
        cosine_min=(0.999891, 0.999276, 0.998394, 0.998493),
        cosine_mean=(0.999947, 0.999766, 0.999432, 0.99936),
    )
    _write_eval_report(
        reports_dir / "eval_imagenette1000_r518_fp32_vs_bf16_prefer.json",
        image_size=518,
        cosine_min=(0.999868, 0.999075, 0.998604, 0.999171),
        cosine_mean=(0.999945, 0.999800, 0.999655, 0.999721),
    )

    manifest = build_cosine_figures(reports_dir, output_dir)

    assert (output_dir / "benchmark_bf16_cosine_min.svg").exists()
    assert (output_dir / "benchmark_bf16_cosine_mean.svg").exists()
    assert (output_dir / "cosine_figures_manifest.json").exists()
    assert {fig["name"] for fig in manifest["figures"]} == {
        "bf16-prefer-cosine-min",
        "bf16-prefer-cosine-mean",
    }
    for fig in manifest["figures"]:
        assert fig["row_count"] == 12
        assert fig["missing_reports"] == []


def _write_speedup_report(
    path: Path,
    *,
    rows: list[dict[str, float]],
    metric_name: str = "gpu_compute_time",
) -> None:
    payload = {
        "reference": "fp32",
        "candidate": "candidate",
        "metric_name": metric_name,
        "shared_batches": [int(row["batch_size"]) for row in rows],
        "reference_only_batches": [],
        "candidate_only_batches": [],
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_default_tradeoff_specs_cover_bf16_and_int8() -> None:
    spec = DEFAULT_TRADEOFF_FIGURE_SPECS[0]
    candidates = {point.candidate for point in spec.points}
    assert "BF16 prefer" in candidates
    assert "INT8 layers16-19" in candidates
    assert "INT8 layer19" in candidates
    assert "INT8 layer19_attention" in candidates
    assert "INT8 SmoothQuant α=0.8 mixed l16-19:fp32" in candidates
    assert "INT8 SmoothQuant α=0.8 ONNX-stripped l16-19 (V1.2)" in candidates
    assert spec.batch_size == 8


def test_extract_tradeoff_points_pulls_cosine_and_speedup(tmp_path: Path) -> None:
    _write_eval_report(
        tmp_path / "eval_bf16.json",
        image_size=224,
        cosine_min=(0.999933, 0.999664, 0.998943, 0.998749),
        cosine_mean=(0.999953, 0.999788, 0.999377, 0.999127),
    )
    _write_speedup_report(
        tmp_path / "speed_bf16.json",
        rows=[
            {
                "batch_size": 8,
                "reference_median_ms": 28.322,
                "candidate_median_ms": 10.075,
                "latency_speedup": 2.81,
                "reference_throughput_qps": 282.0,
                "candidate_throughput_qps": 793.0,
                "throughput_speedup": 2.81,
            },
            {
                "batch_size": 1,
                "reference_median_ms": 7.04,
                "candidate_median_ms": 2.87,
                "latency_speedup": 2.45,
                "reference_throughput_qps": 142.0,
                "candidate_throughput_qps": 348.0,
                "throughput_speedup": 2.45,
            },
        ],
    )
    _write_eval_report(
        tmp_path / "eval_int8.json",
        image_size=224,
        cosine_min=(0.999998, 0.999996, 0.999990, 0.988792),
        cosine_mean=(0.999999, 0.999998, 0.999997, 0.989177),
    )
    _write_speedup_report(
        tmp_path / "speed_int8.json",
        rows=[
            {
                "batch_size": 8,
                "reference_median_ms": 28.322,
                "candidate_median_ms": 23.215,
                "latency_speedup": 1.22,
                "reference_throughput_qps": 282.0,
                "candidate_throughput_qps": 344.0,
                "throughput_speedup": 1.22,
            },
        ],
    )

    spec = TradeoffFigureSpec(
        name="bf16-vs-int8",
        title="BF16 vs INT8",
        points=(
            TradeoffPoint(
                candidate="BF16 prefer",
                eval_filename="eval_bf16.json",
                speedup_filename="speed_bf16.json",
            ),
            TradeoffPoint(
                candidate="INT8 layers16-19",
                eval_filename="eval_int8.json",
                speedup_filename="speed_int8.json",
            ),
        ),
        output_filename="tradeoff.svg",
        batch_size=8,
        cosine_metric="cosine_mean",
    )
    points, missing = extract_tradeoff_points(spec, tmp_path)

    assert missing == []
    by_name = {point.candidate: point for point in points}
    assert by_name["BF16 prefer"].cosine == pytest.approx(0.999127)
    assert by_name["BF16 prefer"].latency_speedup == pytest.approx(2.81)
    assert by_name["INT8 layers16-19"].cosine == pytest.approx(0.989177)
    assert by_name["INT8 layers16-19"].latency_speedup == pytest.approx(1.22)


def test_extract_tradeoff_points_raises_when_speedup_missing(tmp_path: Path) -> None:
    _write_eval_report(
        tmp_path / "eval.json",
        image_size=224,
        cosine_min=(0.999, 0.999, 0.999, 0.999),
        cosine_mean=(0.999, 0.999, 0.999, 0.999),
    )
    spec = TradeoffFigureSpec(
        name="bf16-vs-int8",
        title="t",
        points=(
            TradeoffPoint(
                candidate="X",
                eval_filename="eval.json",
                speedup_filename="missing_speedup.json",
            ),
        ),
        output_filename="t.svg",
    )
    with pytest.raises(FileNotFoundError, match="missing_speedup.json"):
        extract_tradeoff_points(spec, tmp_path)


def test_render_tradeoff_svg_marks_g2_thresholds_and_ideal_region() -> None:
    spec = TradeoffFigureSpec(
        name="t",
        title="cosine vs speedup",
        points=(
            TradeoffPoint(
                candidate="BF16",
                eval_filename="eval.json",
                speedup_filename="speed.json",
                color="#16a34a",
            ),
            TradeoffPoint(
                candidate="INT8",
                eval_filename="eval2.json",
                speedup_filename="speed2.json",
                color="#dc2626",
            ),
        ),
        output_filename="t.svg",
        batch_size=8,
        cosine_metric="cosine_mean",
    )
    from dinov3_trt.reports.benchmark_figures import TradeoffPlotPoint

    points = [
        TradeoffPlotPoint(
            candidate="BF16",
            cosine=0.999127,
            latency_speedup=2.81,
            color="#16a34a",
            annotate=True,
        ),
        TradeoffPlotPoint(
            candidate="INT8 layers16-19",
            cosine=0.989177,
            latency_speedup=1.22,
            color="#dc2626",
            annotate=True,
        ),
    ]
    svg = render_tradeoff_svg(spec, points)

    assert svg.startswith("<svg")
    assert "cosine vs speedup" in svg
    # Thresholds present
    assert "cos = 0.99" in svg
    assert "speedup = 2.2×" in svg
    # Ideal region rectangle is rendered (class is unique to this figure)
    assert "ideal-region" in svg
    # Each point's annotation appears
    assert "BF16" in svg
    assert "INT8 layers16-19" in svg
    assert "0.9991" in svg or "0.999127" in svg


def test_build_tradeoff_figures_writes_svg_and_manifest(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    output_dir = tmp_path / "figures"

    _write_eval_report(
        reports_dir / "eval_a.json",
        image_size=224,
        cosine_min=(0.999, 0.999, 0.999, 0.998),
        cosine_mean=(0.9999, 0.9998, 0.9994, 0.9991),
    )
    _write_speedup_report(
        reports_dir / "speed_a.json",
        rows=[
            {
                "batch_size": 8,
                "reference_median_ms": 28.0,
                "candidate_median_ms": 10.0,
                "latency_speedup": 2.8,
                "reference_throughput_qps": 280.0,
                "candidate_throughput_qps": 784.0,
                "throughput_speedup": 2.8,
            },
        ],
    )
    _write_eval_report(
        reports_dir / "eval_b.json",
        image_size=224,
        cosine_min=(0.999, 0.999, 0.999, 0.989),
        cosine_mean=(0.9999, 0.9999, 0.9999, 0.989),
    )
    _write_speedup_report(
        reports_dir / "speed_b.json",
        rows=[
            {
                "batch_size": 8,
                "reference_median_ms": 28.0,
                "candidate_median_ms": 23.0,
                "latency_speedup": 1.22,
                "reference_throughput_qps": 280.0,
                "candidate_throughput_qps": 342.0,
                "throughput_speedup": 1.22,
            },
        ],
    )

    spec = TradeoffFigureSpec(
        name="bf16-vs-int8",
        title="t",
        points=(
            TradeoffPoint(
                candidate="BF16 prefer",
                eval_filename="eval_a.json",
                speedup_filename="speed_a.json",
            ),
            TradeoffPoint(
                candidate="INT8 layers16-19",
                eval_filename="eval_b.json",
                speedup_filename="speed_b.json",
            ),
        ),
        output_filename="benchmark_bf16_vs_int8_tradeoff.svg",
    )
    manifest = build_tradeoff_figures(reports_dir, output_dir, specs=(spec,))

    assert (output_dir / "benchmark_bf16_vs_int8_tradeoff.svg").exists()
    assert (output_dir / "tradeoff_figures_manifest.json").exists()
    assert manifest["figures"][0]["row_count"] == 2
    assert manifest["figures"][0]["missing_reports"] == []


def test_extract_layer_ablation_points_uses_diversity_ranking_order(
    tmp_path: Path,
) -> None:
    report = tmp_path / "ablation.json"
    report.write_text(
        json.dumps(
            {
                "candidates": {
                    "project": {
                        "layer_numbers_1based": [4, 12, 16, 20],
                        "pairwise_cosine_overall_mean": 0.383,
                        "per_output_magnitude_mean": [362.0, 972.0, 1753.0, 4560.0],
                    },
                    "dpt": {
                        "layer_numbers_1based": [5, 11, 17, 23],
                        "pairwise_cosine_overall_mean": 0.299,
                        "per_output_magnitude_mean": [398.0, 833.0, 2137.0, 12704.0],
                    },
                    "late": {
                        "layer_numbers_1based": [6, 12, 18, 24],
                        "pairwise_cosine_overall_mean": 0.339,
                        "per_output_magnitude_mean": [460.0, 972.0, 2692.0, 38652.0],
                    },
                },
                "diversity_ranking_low_to_high_cosine": ["dpt", "late", "project"],
            }
        ),
        encoding="utf-8",
    )

    points = extract_layer_ablation_points(report)

    assert [p.candidate for p in points] == ["dpt", "late", "project"]
    assert points[0].layer_label == "5/11/17/23"
    assert abs(points[0].mean_cosine - 0.299) < 1e-9
    assert abs(points[0].magnitude_max_min_ratio - 12704.0 / 398.0) < 1e-6
    assert abs(points[2].magnitude_max_min_ratio - 4560.0 / 362.0) < 1e-6


def test_extract_layer_ablation_points_falls_back_to_dict_order(tmp_path: Path) -> None:
    report = tmp_path / "ablation_no_ranking.json"
    report.write_text(
        json.dumps(
            {
                "candidates": {
                    "project": {
                        "layer_numbers_1based": [4, 12, 16, 20],
                        "pairwise_cosine_overall_mean": 0.4,
                        "per_output_magnitude_mean": [10.0, 20.0, 30.0, 40.0],
                    },
                    "dpt": {
                        "layer_numbers_1based": [5, 11, 17, 23],
                        "pairwise_cosine_overall_mean": 0.3,
                        "per_output_magnitude_mean": [10.0, 20.0, 30.0, 100.0],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    points = extract_layer_ablation_points(report)

    assert [p.candidate for p in points] == ["project", "dpt"]


def test_extract_layer_ablation_points_rejects_missing_candidates(
    tmp_path: Path,
) -> None:
    report = tmp_path / "no_candidates.json"
    report.write_text(json.dumps({"image_count": 1000}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing 'candidates'"):
        extract_layer_ablation_points(report)


def test_render_layer_ablation_svg_renders_three_distinct_color_codes() -> None:
    points = [
        LayerAblationFigurePoint(
            candidate="dpt",
            layer_label="5/11/17/23",
            mean_cosine=0.299,
            magnitude_max_min_ratio=31.9,
        ),
        LayerAblationFigurePoint(
            candidate="late",
            layer_label="6/12/18/24",
            mean_cosine=0.339,
            magnitude_max_min_ratio=84.0,
        ),
        LayerAblationFigurePoint(
            candidate="project",
            layer_label="4/12/16/20",
            mean_cosine=0.383,
            magnitude_max_min_ratio=12.6,
        ),
    ]

    svg = render_layer_ablation_svg(points)

    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert 'fill="#2563eb"' in svg
    assert 'fill="#059669"' in svg
    assert 'fill="#dc2626"' in svg
    assert "dpt L5/11/17/23" in svg
    assert "late L6/12/18/24" in svg
    assert "project L4/12/16/20" in svg
    assert "log10" in svg


def test_render_layer_ablation_svg_rejects_empty_points() -> None:
    with pytest.raises(ValueError, match="no points"):
        render_layer_ablation_svg([])


def test_default_layer_ablation_specs_target_eval1000_r224_report() -> None:
    assert len(DEFAULT_LAYER_ABLATION_FIGURE_SPECS) == 1
    spec = DEFAULT_LAYER_ABLATION_FIGURE_SPECS[0]
    assert spec.report_filename == "layer_ablation_pytorch_eval1000_r224.json"
    assert spec.output_filename == "layer_ablation_diversity_vs_balance.svg"
    assert spec.name == "layer-ablation-diversity-vs-balance"


def _write_layer_ablation_report(path: Path) -> None:
    payload = {
        "candidates": {
            "project": {
                "layer_numbers_1based": [4, 12, 16, 20],
                "pairwise_cosine_overall_mean": 0.3828,
                "per_output_magnitude_mean": [362.0, 972.0, 1753.0, 4560.0],
            },
            "dpt": {
                "layer_numbers_1based": [5, 11, 17, 23],
                "pairwise_cosine_overall_mean": 0.299,
                "per_output_magnitude_mean": [398.0, 833.0, 2137.0, 12704.0],
            },
        },
        "diversity_ranking_low_to_high_cosine": ["dpt", "project"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_layer_ablation_figures_writes_svg_and_manifest(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    output_dir = tmp_path / "figures"
    spec = LayerAblationFigureSpec(
        name="layer-ablation",
        title="Layer ablation test",
        report_filename="ablation.json",
        output_filename="layer_ablation.svg",
    )
    _write_layer_ablation_report(reports_dir / "ablation.json")

    manifest = build_layer_ablation_figures(reports_dir, output_dir, specs=(spec,))

    svg_path = output_dir / "layer_ablation.svg"
    assert svg_path.exists()
    assert svg_path.read_text(encoding="utf-8").startswith("<svg")
    manifest_path = output_dir / "layer_ablation_figures_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["figures"][0]["name"] == "layer-ablation"
    assert payload["figures"][0]["row_count"] == 2
    assert manifest["figures"][0]["points"][0]["candidate"] == "dpt"


def test_build_layer_ablation_figures_raises_when_report_missing(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    output_dir = tmp_path / "figures"
    spec = LayerAblationFigureSpec(
        name="layer-ablation",
        title="Layer ablation test",
        report_filename="not_there.json",
        output_filename="layer_ablation.svg",
    )

    with pytest.raises(FileNotFoundError, match="not found"):
        build_layer_ablation_figures(reports_dir, output_dir, specs=(spec,))


def test_build_layer_ablation_figures_allow_missing_emits_placeholder(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    output_dir = tmp_path / "figures"
    spec = LayerAblationFigureSpec(
        name="layer-ablation",
        title="Layer ablation test",
        report_filename="not_there.json",
        output_filename="layer_ablation.svg",
    )

    manifest = build_layer_ablation_figures(
        reports_dir,
        output_dir,
        specs=(spec,),
        allow_missing=True,
    )

    assert manifest["figures"][0]["row_count"] == 0
    assert "missing_report" in manifest["figures"][0]
    assert not (output_dir / "layer_ablation.svg").exists()
    assert (output_dir / "layer_ablation_figures_manifest.json").exists()

