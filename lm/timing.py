"""Timing and size measurement with explicit device synchronisation."""
from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path

import torch


def sync(device: torch.device | None) -> None:
    if device is None:
        return
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


class Timer:
    """Context manager measuring wall-clock seconds, synchronising the device
    before start and after end so asynchronous kernels are included."""

    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.elapsed = 0.0

    def __enter__(self):
        sync(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.device)
        self.elapsed = time.perf_counter() - self._t0
        return False


def count_params(model: torch.nn.Module) -> dict:
    """Total / trainable parameter counts plus the split into input embedding,
    output layer and the remaining 'core'. Models expose `.embed` and `.out`."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = sum(p.numel() for p in model.embed.parameters()) if hasattr(model, "embed") else 0
    out = sum(p.numel() for p in model.out.parameters()) if hasattr(model, "out") else 0
    return {
        "total": total,
        "trainable": trainable,
        "embedding": emb,
        "output": out,
        "core": total - emb - out,
    }


def file_bytes(path: str | Path) -> int:
    return os.path.getsize(path)


def hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        info["cpu"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        info["memory_gb"] = round(mem / 2**30, 1)
        info["macos"] = platform.mac_ver()[0]
    except Exception:  # not macOS
        pass
    return info
