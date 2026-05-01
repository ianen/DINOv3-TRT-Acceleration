import json
from pathlib import Path

import numpy as np
import pytest

from dinov3_trt.infer.cpp_parity import make_cpp_parity_input, read_cpp_output_dump


def test_make_cpp_parity_input_matches_sine_pattern() -> None:
    values = make_cpp_parity_input(batch_size=1, image_size=2).ravel()

    assert values.shape == (12,)
    assert values[0] == pytest.approx(np.float32(np.sin(0.017)))
    assert values[1] == pytest.approx(np.float32(np.sin(0.034)))


def test_read_cpp_output_dump_loads_relative_binary_tensors(tmp_path: Path) -> None:
    tensor = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    tensor.tofile(tmp_path / "feat_layer_4.float32.bin")
    manifest = {
        "batch_size": 1,
        "input_shape": [1, 3, 224, 224],
        "outputs": [
            {
                "name": "feat_layer_4",
                "dtype": "float32",
                "shape": [1, 2, 3],
                "path": "feat_layer_4.float32.bin",
                "byte_size": 24,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    outputs = read_cpp_output_dump(manifest_path)

    assert tuple(outputs) == ("feat_layer_4",)
    np.testing.assert_array_equal(outputs["feat_layer_4"], tensor)


def test_read_cpp_output_dump_rejects_size_mismatch(tmp_path: Path) -> None:
    np.arange(5, dtype=np.float32).tofile(tmp_path / "feat_layer_4.float32.bin")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "outputs": [
                    {
                        "name": "feat_layer_4",
                        "dtype": "float32",
                        "shape": [1, 2, 3],
                        "path": "feat_layer_4.float32.bin",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected 6"):
        read_cpp_output_dump(manifest_path)

