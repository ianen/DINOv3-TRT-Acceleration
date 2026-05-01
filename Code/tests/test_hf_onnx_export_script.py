from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _load_export_script() -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "export_hf_dinov3_onnx.py"
    spec = importlib.util.spec_from_file_location("export_hf_dinov3_onnx", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hf_export_module_registers_and_freezes_model_parameters() -> None:
    torch = pytest.importorskip("torch")
    export_script = _load_export_script()

    class TinyHFDinoModel(torch.nn.Module):  # type: ignore[name-defined, misc]
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(num_register_tokens=0)
            self.proj = torch.nn.Linear(3, 3)

    model = TinyHFDinoModel()
    wrapper = export_script.make_hf_export_module(model)

    assert "model.proj.weight" in wrapper.state_dict()
    assert any(parameter.requires_grad for parameter in wrapper.parameters())

    export_script.freeze_module_parameters(wrapper)

    assert all(not parameter.requires_grad for parameter in wrapper.parameters())
