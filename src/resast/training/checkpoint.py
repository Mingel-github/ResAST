"""Compact checkpoints: learned parameters and fitted preprocessing buffers."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import torch

from resast.data.audio import CHECKPOINT, REVISION
from resast.models.classifier import MODEL_FLAGS
from resast.models.frozen_head import DIMENSIONS
from .common import TrainingConfig
from .torch_engine import make_model, restore, snapshot


def save_checkpoint(path, model, name, config, seed, epoch):
    value = {
        "schema": 1, "model": name, "training": asdict(config), "seed": int(seed), "epoch": int(epoch),
        "state": snapshot(model),
        "backbone": {"id": CHECKPOINT, "revision": REVISION} if name in MODEL_FLAGS else None,
    }
    torch.save(value, path)


def load_checkpoint(path, device, *, backbone=None):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value["schema"] != 1 or value["model"] not in (*MODEL_FLAGS, *DIMENSIONS):
        raise ValueError("Unsupported checkpoint")
    name, config = value["model"], TrainingConfig(**value["training"])
    if name in MODEL_FLAGS and value["backbone"] != {"id": CHECKPOINT, "revision": REVISION}:
        raise ValueError("Checkpoint uses a different pretrained backbone")
    # Shapes are reconstructed here; all normalization buffers are restored below.
    dimensions = 768 if name in MODEL_FLAGS else DIMENSIONS[name]
    store = SimpleNamespace(frozen_embeddings=np.zeros((3, dimensions), dtype=np.float32))
    model = make_model(
        name, store, np.arange(3), config, value["seed"], backbone=backbone,
        acoustic=np.zeros((3, 20), dtype=np.float32),
    )
    restore(model, value["state"])
    return model.to(device).eval(), value
