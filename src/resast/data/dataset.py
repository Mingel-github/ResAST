"""Call/segment alignment and identity-grouped sampling."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .acoustic import FEATURE_NAMES


def load_clean_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"analysis_cat_id": str, "published_cat_id": str})
    include = frame["analysis_include"].astype(str).str.lower() == "true"
    clean = frame[include].sort_values("source_path").reset_index(drop=True)
    if clean["filename"].duplicated().any():
        raise ValueError("Duplicate call identifiers")
    if clean.groupby("analysis_cat_id")["age_group_filename"].nunique().max() != 1:
        raise ValueError("An analysis ID has conflicting age-group labels")
    return clean


@dataclass(frozen=True)
class TokenStore:
    tokens: np.ndarray
    call_segment_indices: tuple[np.ndarray, ...]
    frozen_embeddings: np.ndarray
    call_ids: np.ndarray
    cat_ids: np.ndarray
    labels: np.ndarray


def load_token_store(
    tokens_path: str | Path, index_path: str | Path,
    fbank_path: str | Path, embeddings_path: str | Path,
) -> TokenStore:
    tokens = np.load(tokens_path, mmap_mode="r", allow_pickle=False)
    if tokens.ndim != 3 or tokens.shape[1:] != (146, 768) or tokens.dtype != np.float32:
        raise ValueError("Expected float32 block-ten token cache [segments, 146, 768]")
    with np.load(index_path, allow_pickle=False) as index:
        if index["block_output_one_based"].tolist() != [10] or index["patch_token_start"].tolist() != [2]:
            raise ValueError("Token cache layer or special-token boundary differs")
        segment_call_indices = index["segment_call_indices"].astype(np.int64)
        segment_counts = index["segment_counts"].astype(np.int64)
        cache_call_ids = index["call_ids"].astype(str)
    with np.load(fbank_path, allow_pickle=False) as source:
        call_ids = source["call_ids"].astype(str)
        cat_ids = source["cat_ids"].astype(str)
        labels = source["labels"].astype(np.int64)
        if not np.array_equal(segment_call_indices, source["segment_call_indices"]):
            raise ValueError("Token and filterbank segment mappings differ")
        if not np.array_equal(segment_counts, source["segment_counts"]):
            raise ValueError("Token and filterbank segment counts differ")
    with np.load(embeddings_path, allow_pickle=False) as frozen:
        frozen_embeddings = frozen["embeddings"].astype(np.float32)
        frozen_call_ids = frozen["call_ids"].astype(str)
    if not np.array_equal(cache_call_ids, call_ids) or not np.array_equal(frozen_call_ids, call_ids):
        raise ValueError("Call order differs across feature sources")
    if len(set(call_ids)) != len(call_ids) or len(cat_ids) != len(call_ids) or len(labels) != len(call_ids):
        raise ValueError("Invalid call identity or label arrays")
    if len(segment_call_indices) != len(tokens) or frozen_embeddings.shape != (len(call_ids), 768):
        raise ValueError("Feature counts differ")
    if not np.isfinite(frozen_embeddings).all() or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("Invalid frozen embeddings or labels")
    if np.any(segment_call_indices < 0) or np.any(segment_call_indices >= len(call_ids)):
        raise ValueError("Segment mapping refers to an unknown call")
    call_segments = tuple(
        np.flatnonzero(segment_call_indices == index).astype(np.int64)
        for index in range(len(call_ids))
    )
    if not np.array_equal([len(v) for v in call_segments], segment_counts) or np.any(segment_counts < 1):
        raise ValueError("Every call must have its declared segments")
    return TokenStore(tokens, call_segments, frozen_embeddings, call_ids, cat_ids, labels)


def load_acoustic_features(path: str | Path, call_ids: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if tuple(data["feature_names"].astype(str)) != FEATURE_NAMES:
            raise ValueError("Acoustic feature order differs")
        if not np.array_equal(data["call_ids"].astype(str), call_ids.astype(str)):
            raise ValueError("Acoustic and token call order differs")
        features = data["features"].astype(np.float32)
    if features.shape != (len(call_ids), 20):
        raise ValueError("Expected 20 acoustic features per call")
    return features


class DeterministicNoSingletonBatchSampler(Sampler):
    def __init__(self, size: int, batch_size: int, seed: int):
        if size < 2 or batch_size < 2:
            raise ValueError("Training batches need at least two identities")
        self.size, self.batch_size = int(size), int(batch_size)
        self.generator = torch.Generator().manual_seed(int(seed))

    def __iter__(self):
        order = torch.randperm(self.size, generator=self.generator).tolist()
        batches = [
            order[start : start + self.batch_size]
            for start in range(0, self.size, self.batch_size)
        ]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-1].insert(0, batches[-2].pop())
        if any(len(batch) == 1 for batch in batches):
            raise RuntimeError("Batch size produces a singleton identity batch")
        yield from batches

    def __len__(self):
        return math.ceil(self.size / self.batch_size)


class CatTokenDataset(Dataset):
    def __init__(self, store: TokenStore, call_indices: np.ndarray):
        self.store = store
        selected = np.asarray(call_indices, dtype=np.int64)
        if selected.ndim != 1 or len(np.unique(selected)) != len(selected):
            raise ValueError("Call indices must be unique")
        if np.any(selected < 0) or np.any(selected >= len(store.call_ids)):
            raise ValueError("Call index is out of range")
        self.cat_ids = np.asarray(sorted(np.unique(store.cat_ids[selected]).tolist()))
        self.call_indices = []
        for cat_id in self.cat_ids:
            calls = np.sort(selected[store.cat_ids[selected] == cat_id])
            if len(np.unique(store.labels[calls])) != 1:
                raise ValueError(f"Conflicting labels for identity {cat_id}")
            self.call_indices.append(calls)

    def __len__(self):
        return len(self.cat_ids)

    def __getitem__(self, item):
        calls = self.call_indices[item]
        token_rows, mappings = [], []
        for local_call, call_index in enumerate(calls):
            segments = self.store.call_segment_indices[int(call_index)]
            token_rows.append(np.asarray(self.store.tokens[segments], dtype=np.float32))
            mappings.append(np.full(len(segments), local_call, dtype=np.int64))
        return np.concatenate(token_rows), np.concatenate(mappings), calls, str(self.cat_ids[item])


def collate_cats(rows) -> dict:
    tokens, mappings, calls, cats = [], [], [], []
    offset = 0
    for row_tokens, row_mapping, row_calls, cat_id in rows:
        tokens.append(row_tokens)
        mappings.append(row_mapping + offset)
        calls.append(row_calls)
        cats.append(cat_id)
        offset += len(row_calls)
    return {
        "tokens": torch.from_numpy(np.concatenate(tokens).astype(np.float32, copy=False)),
        "segment_to_call": torch.from_numpy(np.concatenate(mappings)),
        "call_indices": torch.from_numpy(np.concatenate(calls).astype(np.int64, copy=False)),
        "cat_ids": cats,
    }


def build_loader(store: TokenStore, indices: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    dataset = CatTokenDataset(store, indices)
    if shuffle:
        return DataLoader(
            dataset,
            batch_sampler=DeterministicNoSingletonBatchSampler(len(dataset), batch_size, seed),
            num_workers=0, collate_fn=collate_cats,
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_cats)


def role_indices(store: TokenStore, roles: pd.DataFrame, repeat: int, fold: int) -> dict:
    cell = roles[(roles["repeat"] == repeat) & (roles["outer_fold"] == fold)]
    if cell["cat_id"].duplicated().any():
        raise ValueError("An identity has multiple roles in one fold")
    mapping = dict(zip(cell["cat_id"].astype(str), cell["role"].astype(str)))
    if set(mapping) != set(store.cat_ids.astype(str)):
        raise ValueError("Role table must cover every identity")
    call_roles = np.asarray([mapping[cat_id] for cat_id in store.cat_ids], dtype=str)
    result = {
        role: np.flatnonzero(call_roles == role).astype(np.int64)
        for role in ("train", "validation", "test")
    }
    if sum(map(len, result.values())) != len(store.call_ids):
        raise ValueError("Unknown or missing role")
    return result


def load_roles(path: str | Path) -> pd.DataFrame:
    """Select the paper's repeats from the historical role bank."""
    roles = pd.read_csv(path, dtype={"cat_id": str})
    return roles[roles["repeat"].isin((0, 1, 2))].reset_index(drop=True)


def validate_roles(store: TokenStore, roles: pd.DataFrame) -> int:
    roles = roles[roles["repeat"].isin((0, 1, 2))]
    if set(roles["role"].astype(str)) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation and test roles")
    coordinates = set(zip(roles["repeat"], roles["outer_fold"]))
    if coordinates != {(r, f) for r in range(3) for f in range(4)}:
        raise ValueError("Expected three repeats and four folds")
    for repeat in range(3):
        test_ids = []
        for fold in range(4):
            indices = role_indices(store, roles, repeat, fold)
            if any(len(values) == 0 for values in indices.values()):
                raise ValueError("Empty data role")
            test_ids.extend(np.unique(store.cat_ids[indices["test"]]).tolist())
        if len(test_ids) != len(set(test_ids)) or set(test_ids) != set(store.cat_ids):
            raise ValueError("Every identity must be tested once per repeat")
    return 12
