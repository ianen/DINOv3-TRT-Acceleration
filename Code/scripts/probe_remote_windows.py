#!/usr/bin/env python
"""Probe the RTX 5080 Windows workstation over SSH."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from typing import Dict, List


DEFAULT_PROJECT_DIR = r"D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration"


@dataclass(frozen=True)
class Probe:
    name: str
    command: str


PROBES = (
    Probe(
        "gpu",
        "nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv",
    ),
    Probe("python", "python --version"),
    Probe(
        "torch",
        'python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"',
    ),
    Probe(
        "tensorrt",
        'python -c "import tensorrt as trt; print(trt.__version__)"',
    ),
    Probe("trtexec_path", "where trtexec"),
)


def run_ssh(host: str, command: str, timeout: int) -> Dict[str, object]:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def project_dir_probe(project_dir: str) -> Probe:
    command = f'powershell -NoProfile -Command "Test-Path -LiteralPath \'{project_dir}\'"'
    return Probe("project_dir_exists", command)


def collect(host: str, project_dir: str, timeout: int) -> Dict[str, object]:
    probes: List[Probe] = list(PROBES) + [project_dir_probe(project_dir)]
    return {
        "host": host,
        "project_dir": project_dir,
        "probes": {probe.name: run_ssh(host, probe.command, timeout) for probe in probes},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="windows-pc")
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    print(json.dumps(collect(args.host, args.project_dir, args.timeout), indent=2))


if __name__ == "__main__":
    main()
