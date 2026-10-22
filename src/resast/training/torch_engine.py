"""PyTorch fitting for AST variants and frozen-embedding baselines."""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from resast.data.dataset import build_loader
from resast.data.features import build_feature_loader
from resast.evaluation.metrics import PROBABILITIES, metrics, units_to_cats
from resast.models.classifier import MODEL_FLAGS, build_classifier
from resast.models.frozen_head import FrozenEmbeddingHead
from .common import check_fit, class_weights, set_seed, state_copy


@dataclass
class FitResult:
    model: object
    best_epoch: int
    history: list
    cats: pd.DataFrame | None
    units: pd.DataFrame | None


def snapshot(model):
    if hasattr(model, "tail"):
        # Store the learned head and gates; copied AST tail weights stay frozen.
        return {
            "head": state_copy(model.head),
            "raw_gamma": None if model.tail.raw_gamma is None else model.tail.raw_gamma.detach().cpu().clone(),
        }
    return {"head": state_copy(model)}


def restore(model, state):
    if hasattr(model, "tail"):
        model.head.load_state_dict(state["head"], strict=True)
        gamma = state["raw_gamma"]
        if (gamma is None) != (model.tail.raw_gamma is None):
            raise ValueError("Checkpoint multiscale structure differs")
        if gamma is not None:
            with torch.no_grad():
                model.tail.raw_gamma.copy_(gamma)
    else:
        model.load_state_dict(state["head"], strict=True)


def make_model(name, store, train, config, seed, backbone=None, acoustic=None):
    set_seed(seed)
    if name in MODEL_FLAGS:
        if backbone is None:
            raise ValueError("AST variants require the frozen backbone")
        return build_classifier(
            name, backbone, store.frozen_embeddings[train], dropout=config.dropout,
            acoustic_train=None if acoustic is None else acoustic[train],
        )
    return FrozenEmbeddingHead(store.frozen_embeddings[train], config.dropout)


def loader_for(model, store, indices, batch_size, shuffle, seed):
    factory = build_loader if hasattr(model, "tail") else build_feature_loader
    return factory(store, indices, batch_size, shuffle, seed)


def batch_logits(model, batch, acoustic, device):
    if hasattr(model, "tail"):
        indices = batch["call_indices"].numpy()
        age = None
        if MODEL_FLAGS[model.model_name][1]:
            if acoustic is None:
                raise ValueError("Acoustic residual requires aligned features")
            age = torch.from_numpy(acoustic[indices]).to(device)
        return model(
            batch["tokens"].to(device), batch["segment_to_call"].to(device), len(indices), age,
        )
    return model(batch["embeddings"].to(device))


def train_epoch(model, loader, optimizer, scaler, store, acoustic, weights, device, gradient_clip):
    model.train()
    batch_losses, weighted_total, weight_total = [], 0.0, 0.0
    seen_cats, seen_calls = [], []
    parameters = [p for p in model.parameters() if p.requires_grad]
    for batch in loader:
        indices = batch["call_indices"].numpy()
        labels = torch.from_numpy(store.labels[indices]).to(device)
        # Apply inverse-frequency class weights per call on the training side only.
        call_weights = weights[labels]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = batch_logits(model, batch, acoustic, device)
            per_call = nn.functional.cross_entropy(logits, labels, reduction="none")
            loss = (per_call * call_weights).sum() / call_weights.sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(parameters, gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        batch_losses.append(float(loss.detach()))
        weighted_total += float((per_call.detach() * call_weights).sum())
        weight_total += float(call_weights.sum())
        seen_cats.extend(batch["cat_ids"])
        seen_calls.extend(indices.tolist())
    if len(set(seen_cats)) != len(seen_cats) or len(set(seen_calls)) != len(seen_calls):
        raise RuntimeError("Training epoch repeated an identity or call")
    # Historical paths differ only in their training-loss reporting reduction.
    return weighted_total / weight_total if hasattr(model, "tail") else float(np.mean(batch_losses))


def predict(model, store, indices, config, device, seed, acoustic=None):
    loader = loader_for(model, store, indices, config.batch_size * 2, False, seed)
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = batch_logits(model, batch, acoustic, device)
                if not hasattr(model, "tail"):
                    probabilities = torch.softmax(logits, dim=1).float().cpu().numpy()
            if hasattr(model, "tail"):
                probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            for local, index in enumerate(batch["call_indices"].numpy()):
                rows.append({
                    "call_index": int(index), "unit_id": str(store.call_ids[index]),
                    "cat_id": str(store.cat_ids[index]), "true_label": int(store.labels[index]),
                    **{name: float(probabilities[local, i]) for i, name in enumerate(PROBABILITIES)},
                })
    units = pd.DataFrame(rows).sort_values("call_index").reset_index(drop=True)
    return units_to_cats(units), units


def fit(
    name, store, train, config, device, seed, *, validation=None, fixed_epochs=None,
    backbone=None, acoustic=None, on_epoch=None,
):
    check_fit(store, train, validation, fixed_epochs, config)
    model = make_model(name, store, train, config, seed, backbone, acoustic).to(device)
    # Reset training RNGs after model initialization.
    set_seed(seed + 1_000_000)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adamax(
        parameters, lr=config.learning_rate, eps=config.optimizer_epsilon, weight_decay=config.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loader = loader_for(model, store, train, config.batch_size, True, seed)
    weights = torch.from_numpy(class_weights(store.labels[train])).to(device)
    best_loss, best_epoch, stale = math.inf, 0, 0
    best_state, best_cats, best_units, history = None, None, None, []
    maximum = config.maximum_epochs if fixed_epochs is None else fixed_epochs
    for epoch in range(1, maximum + 1):
        loss = train_epoch(model, loader, optimizer, scaler, store, acoustic, weights, device, config.gradient_clip)
        row = {"epoch": epoch, "training_call_loss": loss}
        if validation is not None:
            cats, units = predict(model, store, validation, config, device, seed, acoustic)
            row["validation_metrics"] = metrics(cats)
            # Select epochs by unweighted identity-level validation cross-entropy.
            value = row["validation_metrics"]["cross_entropy"]
            if value < best_loss - config.minimum_delta:
                best_loss, best_epoch, stale = value, epoch, 0
                best_state, best_cats, best_units = snapshot(model), cats.copy(), units.copy()
            else:
                stale += 1
        history.append(row)
        if on_epoch is not None:
            on_epoch(row)
        if validation is not None and stale >= config.patience:
            break
    if validation is not None:
        if best_state is None:
            raise RuntimeError("No finite validation checkpoint")
        restore(model, best_state)
    else:
        best_epoch = maximum
    return FitResult(model, best_epoch, history, best_cats, best_units)
