"""Validated, frozen feature tables and identity-grouped loading."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .dataset import DeterministicNoSingletonBatchSampler


@dataclass(frozen=True)
class FeatureStore:
    frozen_embeddings: np.ndarray
    call_ids: np.ndarray
    cat_ids: np.ndarray
    labels: np.ndarray


def load_features(path, dimensions: int) -> FeatureStore:
    """Load call embeddings or author feature rows, preserving their stored order."""
    with np.load(path, allow_pickle=False) as data:
        store = FeatureStore(
            data["embeddings"].astype(np.float32), data["call_ids"].astype(str),
            data["cat_ids"].astype(str), data["labels"].astype(np.int64),
        )
        if not np.array_equal(data["labels"], store.labels):
            raise ValueError("Labels must be integers")
    n = len(store.labels)
    if n == 0 or store.frozen_embeddings.shape != (n, dimensions):
        raise ValueError("Feature table dimensions differ")
    if store.cat_ids.shape != (n,) or store.call_ids.shape != (n,) or len(set(store.call_ids)) != n:
        raise ValueError("Invalid feature unit identity arrays")
    if not np.isfinite(store.frozen_embeddings).all() or not np.isin(store.labels, (0, 1, 2)).all():
        raise ValueError("Invalid embeddings or labels")
    for cat_id in np.unique(store.cat_ids):
        if len(np.unique(store.labels[store.cat_ids == cat_id])) != 1:
            raise ValueError("Conflicting age-group labels within an identity")
    return store


def load_author_features(path, name, analysis_ids) -> FeatureStore:
    dimensions = {"vggish129": 128, "perch1281": 1280, "yamnet": 1024}[name]
    frame = pd.read_csv(path, dtype={"cat_id": str})
    if name == "vggish129":
        clean = frame[frame.cat_id.isin(analysis_ids)].reset_index(drop=True)
        units = np.asarray([f"vggish-row-{i:04d}" for i in range(len(clean))])
    else:
        if (frame.cat_id == "049A").sum() != 1:
            raise ValueError("Expected one excluded duplicate author row")
        clean = frame[frame.cat_id != "049A"].reset_index(drop=False)
        prefix = "Perch1281" if name == "perch1281" else "YAMNet1024"
        units = np.asarray([f"{prefix}_source_row_{i:06d}" for i in clean["index"]])
    expected = {"vggish129": 936, "perch1281": 792, "yamnet": 967}[name]
    if len(clean) != expected or set(clean.cat_id) != set(analysis_ids):
        raise ValueError("Author features differ from the clean analysis cohort")
    columns = [str(i) for i in range(dimensions)] + ([] if name == "yamnet" else ["mean_freq"])
    features = clean[columns].to_numpy(dtype=np.float32)
    age = clean.target.to_numpy(dtype=np.float64)
    if not np.isfinite(features).all() or not np.all((age >= 0) & (age < 20)):
        raise ValueError("Invalid author features or ages")
    labels = np.where(age < .5, 0, np.where(age < 10, 1, 2)).astype(np.int64)
    return FeatureStore(features, units, clean.cat_id.to_numpy(str), labels)


class CatFeatureDataset(Dataset):
    def __init__(self, store, indices):
        self.store = store
        self.cat_ids = sorted(np.unique(store.cat_ids[indices]).tolist())
        self.calls = [np.sort(indices[store.cat_ids[indices] == cat]) for cat in self.cat_ids]

    def __len__(self):
        return len(self.calls)

    def __getitem__(self, item):
        calls = self.calls[item]
        return self.store.frozen_embeddings[calls], calls, self.cat_ids[item]


def collate_features(rows):
    return {
        "embeddings": torch.from_numpy(np.concatenate([r[0] for r in rows])),
        "call_indices": torch.from_numpy(np.concatenate([r[1] for r in rows])),
        "cat_ids": [r[2] for r in rows],
    }


def build_feature_loader(store, indices, batch_size, shuffle, seed):
    dataset = CatFeatureDataset(store, indices)
    if shuffle:
        return DataLoader(
            dataset, batch_sampler=DeterministicNoSingletonBatchSampler(len(dataset), batch_size, seed),
            num_workers=0, collate_fn=collate_features,
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_features)
