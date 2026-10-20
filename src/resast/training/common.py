"""Shared numerical and training controls."""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.006
    dropout: float = 0.44571035356880917
    weight_decay: float = 0.0
    optimizer_epsilon: float = 1e-7
    batch_size: int = 4
    maximum_epochs: int = 50
    patience: int = 8
    minimum_delta: float = 1e-6
    gradient_clip: float = 1.0

    def __post_init__(self):
        numeric = np.asarray(list(asdict(self).values()), dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("Training settings must be finite")
        if self.learning_rate <= 0 or not 0 <= self.dropout < 1 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer or dropout setting")
        if self.optimizer_epsilon <= 0 or self.minimum_delta < 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid numerical tolerance")
        for value in (self.batch_size, self.maximum_epochs, self.patience):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Batch size, epochs and patience must be positive integers")


def full_seed(base_seed: int, repeat: int, fold: int) -> int:
    return int(base_seed) + 10_000 * int(repeat) + 100 * int(fold)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int64), minlength=3).astype(np.float64)
    if len(counts) != 3 or np.any(counts == 0):
        raise ValueError("The training side must contain all three age groups")
    return (len(labels) / (3 * counts)).astype(np.float32)


def standardizer(values):
    mean, scale = values.mean(axis=0).astype(np.float32), values.std(axis=0).astype(np.float32)
    return mean, np.where(scale > 1e-12, scale, 1).astype(np.float32)


def check_fit(store, train, validation, fixed_epochs, cfg):
    if (validation is None) == (fixed_epochs is None):
        raise ValueError("Choose validation-based selection or fixed-epoch refitting")
    for indices in (train,) if validation is None else (train, validation):
        if len(indices) == 0 or len(np.unique(indices)) != len(indices):
            raise ValueError("Fit indices must be nonempty and unique")
        if np.any(indices < 0) or np.any(indices >= len(store.labels)):
            raise ValueError("Fit index is out of range")
    if validation is not None and set(store.cat_ids[train]) & set(store.cat_ids[validation]):
        raise ValueError("Training and validation identities overlap")
    if fixed_epochs is not None and (
        isinstance(fixed_epochs, bool) or not isinstance(fixed_epochs, int)
        or not 1 <= fixed_epochs <= cfg.maximum_epochs
    ):
        raise ValueError("Fixed epochs exceed the declared training budget")


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def state_copy(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
