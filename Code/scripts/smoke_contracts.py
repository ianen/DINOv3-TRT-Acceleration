#!/usr/bin/env python
"""Run lightweight contract checks without installing the package."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_trt.contracts import (  # noqa: E402
    DINO_VITL16_224_CONTRACT,
    expected_output_shape,
    expected_token_count,
)


def main() -> None:
    contract = DINO_VITL16_224_CONTRACT
    result = {
        "model_id": contract.model_id,
        "layer_indices": contract.layer_indices,
        "layer_numbers": contract.layer_numbers,
        "output_names": contract.output_names,
        "main_tokens": expected_token_count(),
        "with_register_tokens": expected_token_count(include_register_tokens=True),
        "batch2_shape": expected_output_shape(2),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
