"""Shared training utilities: seeding, schedules, logging, checkpoints."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 0.0) -> float:
    """LR multiplier: linear warmup to 1, then cosine decay to ``min_ratio``."""

    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def warmup_constant(step: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    return 1.0


def decay_parameter_groups(model, weight_decay: float):
    """AdamW groups: no weight decay on biases and normalization weights."""

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name.lower() or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class JsonlLogger:
    """Append-only metrics file, flushed after every record."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.time()

    def log(self, record: dict) -> None:
        record = {**record, "elapsed_sec": round(time.time() - self.start, 1)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def save_checkpoint(model, tokenizer, out_dir: str | Path, extra: dict | None = None) -> None:
    """Write model + tokenizer atomically (tmp dir, then rename)."""

    out_dir = Path(out_dir)
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    model.save_pretrained(tmp_dir, safe_serialization=True)
    tokenizer.save_pretrained(tmp_dir)
    if extra is not None:
        (tmp_dir / "checkpoint_info.json").write_text(json.dumps(extra, indent=2))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)


def environment_info() -> dict:
    import transformers

    info = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    try:
        import datasets

        info["datasets"] = datasets.__version__
    except ImportError:
        pass
    return info


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
