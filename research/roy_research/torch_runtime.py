from __future__ import annotations

import os
from typing import Dict

import torch


_CONFIGURED = False


def configure_torch_runtime() -> Dict[str, int]:
    """Bound per-sidecar CPU pools before any model work begins."""
    global _CONFIGURED
    threads = max(1, int(os.environ.get("ROY_LHTB_TORCH_THREADS", "4")))
    interop = max(1, int(os.environ.get("ROY_LHTB_TORCH_INTEROP_THREADS", "1")))
    if not _CONFIGURED:
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(interop)
        except RuntimeError:
            if torch.get_num_interop_threads() != interop:
                raise
        _CONFIGURED = True
    return {
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }
