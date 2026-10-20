import numpy as np
import pandas as pd
import pytest
import torch

from resast.data.audio import segment_waveform, average_frozen_segments
from resast.data.acoustic import FEATURE_NAMES, extract_call_features
from resast.data.dataset import (
    TokenStore, CatTokenDataset, collate_cats,
    DeterministicNoSingletonBatchSampler, role_indices, load_clean_manifest,
    load_roles,
)


@pytest.mark.parametrize("length,count", [(800, 1), (20480, 1), (20481, 2), (40960, 3)])
def test_waveform_windows(length, count):
    waveform = np.arange(length, dtype=np.float32)
    segments = segment_waveform(waveform)
    assert len(segments) == count
    assert segments[-1][-1] == waveform[-1]
    assert all(len(x) <= 20480 for x in segments)


def test_empty_waveform_rejected():
    with pytest.raises(ValueError):
        segment_waveform(np.zeros(0, dtype=np.float32))


def test_frozen_embedding_mean_uses_float64_accumulation():
    segments = np.array([[1e8], [1.0], [-1e8]], dtype=np.float32)
    result = average_frozen_segments(segments, np.array([0, 0, 0]), np.array([3]))
    np.testing.assert_array_equal(result, np.array([[1 / 3]], dtype=np.float32))
    with pytest.raises(ValueError, match="counts"):
        average_frozen_segments(segments, np.array([0, 0, 0]), np.array([2]))


@pytest.mark.parametrize("size", [4, 5, 9, 13])
def test_batches_are_reproducible_and_cover_every_identity(size):
    left = DeterministicNoSingletonBatchSampler(size, 4, 17)
    right = DeterministicNoSingletonBatchSampler(size, 4, 17)
    for _ in range(3):
        batches = list(left)
        assert batches == list(right)
        assert sorted(i for batch in batches for i in batch) == list(range(size))
        assert all(len(batch) > 1 for batch in batches)


def make_store():
    return TokenStore(
        np.zeros((4, 146, 768), np.float32),
        (np.array([0, 1]), np.array([2]), np.array([3])),
        np.zeros((3, 768), np.float32),
        np.array(["call0", "call1", "call2"]),
        np.array(["b", "a", "b"]),
        np.array([0, 1, 0]),
    )


def test_grouped_calls_and_local_segment_mapping():
    dataset = CatTokenDataset(make_store(), np.array([0, 1, 2]))
    batch = collate_cats([dataset[0], dataset[1]])
    assert batch["cat_ids"] == ["a", "b"]
    torch.testing.assert_close(batch["call_indices"], torch.tensor([1, 0, 2]))
    torch.testing.assert_close(batch["segment_to_call"], torch.tensor([0, 1, 1, 2]))


def test_role_table_keeps_every_call_of_an_identity_together():
    roles = pd.DataFrame({
        "repeat": [0, 0], "outer_fold": [0, 0],
        "cat_id": ["a", "b"], "role": ["test", "train"],
    })
    indices = role_indices(make_store(), roles, 0, 0)
    np.testing.assert_array_equal(indices["train"], [0, 2])
    np.testing.assert_array_equal(indices["test"], [1])
    with pytest.raises(ValueError, match="multiple roles"):
        role_indices(make_store(), pd.concat([roles, roles.iloc[:1]]), 0, 0)


def test_manifest_filter_preserves_ids_and_source_order(tmp_path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame({
        "filename": ["b.wav", "a.wav", "dup.wav"],
        "source_path": ["b", "a", "dup"],
        "analysis_include": ["true", "true", "false"],
        "analysis_cat_id": ["001A", "002A", "003A"],
        "published_cat_id": ["001A", "002A", "003A"],
        "age_group_filename": ["adult", "kitten", "adult"],
    }).to_csv(path, index=False)
    clean = load_clean_manifest(path)
    assert clean["filename"].tolist() == ["a.wav", "b.wav"]
    assert clean["analysis_cat_id"].tolist() == ["002A", "001A"]


def test_paper_roles_select_three_repeats_from_larger_bank(tmp_path):
    path = tmp_path / "roles.csv"
    pd.DataFrame({"repeat": [0, 1, 2, 3, 4], "cat_id": ["001A"] * 5}).to_csv(path, index=False)
    selected = load_roles(path)
    assert selected["repeat"].tolist() == [0, 1, 2]
    assert selected["cat_id"].tolist() == ["001A"] * 3


def test_acoustic_descriptors_have_fixed_order_and_missing_pitch_policy():
    assert len(FEATURE_NAMES) == 20
    features = extract_call_features(np.zeros(3200, np.float32))
    assert features.shape == (20,)
    assert features.dtype == np.float32
    assert np.isnan(features[0])
    assert features[FEATURE_NAMES.index("voiced_fraction")] == 0
