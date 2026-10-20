import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from resast.cli import extract, prepare, reference, search
from resast.cli.train import cell_config
from resast.data.dataset import load_clean_manifest
from resast.training.common import sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entry", [extract.main, prepare.main, reference.main, search.main])
def test_release_entry_help(entry):
    with pytest.raises(SystemExit) as error:
        entry(["--help"])
    assert error.value.code == 0


def test_paper_configs_and_epoch_coverage():
    config = json.loads((ROOT / "configs/paper.json").read_text())
    assert config["seeds"] == [17, 43, 101]
    assert len(config["models"]) == 10
    cells = []
    for model in config["models"]:
        for repeat in range(3):
            for fold in range(4):
                cfg = cell_config(config, model, repeat, fold)
                assert cfg.learning_rate > 0
                cells.extend((model, repeat, fold, seed) for seed in config["seeds"])
    epochs = pd.read_csv(ROOT / "configs/reference_epochs.csv")
    assert set(epochs[["model", "repeat", "fold", "base_seed"]].itertuples(index=False, name=None)) == set(cells)
    assert len(epochs) == 360 and epochs.epoch.min() >= 1


def test_role_cohort_and_metadata():
    clean = load_clean_manifest(ROOT / "data/manifest.csv")
    roles = pd.read_csv(ROOT / "splits/roles.csv", dtype={"cat_id": str})
    assert len(clean) == 792 and clean.analysis_cat_id.nunique() == 111
    assert len(roles) == 1332
    for repeat in range(3):
        subset = roles[roles.repeat == repeat]
        assert set(subset[subset.role == "test"].cat_id) == set(clean.analysis_cat_id)
        assert not subset[subset.role == "test"].cat_id.duplicated().any()
        for fold in range(4):
            cell = subset[subset.outer_fold == fold]
            assert len(cell) == 111 and not cell.cat_id.duplicated().any()
            assert set(cell.role) == {"train", "validation", "test"}
    prov = json.loads((ROOT / "data/provenance.json").read_text())
    assert sha256(ROOT / "data/manifest.csv") == prov["manifest_sha256"]
    assert sha256(ROOT / "splits/roles.csv") == prov["roles_sha256"]


def test_released_predictions_and_table():
    result = reference.verify(ROOT)
    assert result["prediction_files"] == 120 and result["models"] == 10
    assert len(result["rows"]) == 160


def test_search_ranking_matches_all_locked_cells():
    spec = json.loads((ROOT / "configs/search.json").read_text())
    config = json.loads((ROOT / "configs/paper.json").read_text())
    for entry in spec["selected"]:
        candidates = [{"configuration_id": c["config_id"], **{
            k: c[k] for k in ("mean_macro_f1", "mean_accuracy", "mean_brier")}}
            for c in entry["candidates"]]
        assert search.choose(candidates)["configuration_id"] == entry["configuration_id"]
        cell = [c for c in config["models"][entry["model"]]["cells"]
                if c["repeat"] == entry["repeat"] and c["fold"] == entry["fold"]][0]
        assert cell["configuration_id"] == entry["configuration_id"]
    assert len(spec["selected"]) == 48
    with pytest.raises(ValueError):
        search.choose([])


def test_search_cli_uses_validation_only(tmp_path, monkeypatch):
    from test_evaluation import predictions
    roles = pd.DataFrame({"cat_id": ["a", "b", "c"], "repeat": [0]*3, "outer_fold": [0]*3,
                          "role": ["train", "validation", "test"]})
    store = SimpleNamespace(cat_ids=np.array(["a", "b", "c"]))
    monkeypatch.setattr(search, "load_inputs", lambda *a: (store, roles, None, None, {}))
    monkeypatch.setattr(search, "role_indices", lambda *a: {
        "train": np.array([0]), "validation": np.array([1]), "test": np.array([2])})
    calls = []
    def fake_fit(name, store, indices, cfg, device, seed, **kwargs):
        assert indices.tolist() == [0] and kwargs["validation"].tolist() == [1]
        calls.append(seed)
        return SimpleNamespace(cats=predictions(), best_epoch=1, history=[])
    monkeypatch.setattr(search, "fit", fake_fit)
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"seeds": [1090450653, 846528492], "tie_tolerance": 1e-12,
                                "grid": [{"configuration_id": "q00", "learning_rate": .006}]}))
    search.main(["--config", str(ROOT / "configs/paper.json"), "--grid", str(grid),
                 "--model", "ast", "--repeat", "0", "--fold", "0",
                 "--output", str(tmp_path / "search"), "--device", "cpu"])
    assert calls == [1090450653, 846528492]
    assert json.loads((tmp_path / "search/selection.json").read_text())["test_prediction_calls"] == 0


def test_ast_cache_geometry_and_alignment(tmp_path, monkeypatch):
    rows = pd.DataFrame({"filename": ["a.wav", "b.wav"], "analysis_cat_id": ["a", "b"],
                         "age_group_filename": ["kitten", "adult"]})
    data = {
        "features": np.zeros((3, 128, 128), np.float32),
        "segment_call_indices": np.array([0, 0, 1]), "segment_counts": np.array([2, 1]),
        "call_ids": np.array(["a.wav", "b.wav"]), "source_paths": np.array(["a.wav", "b.wav"]),
    }
    class Backbone:
        def to(self, device):
            return self
        def __call__(self, **kwargs):
            return SimpleNamespace(pooler_output=torch.ones((len(kwargs["input_values"]), 768)))
    class Prefix(Backbone):
        def __init__(self, backbone):
            pass
        def __call__(self, batch):
            return torch.zeros((len(batch), 146, 768), dtype=torch.float32)
    monkeypatch.setattr(extract, "make_feature_extractor", lambda *a: None)
    monkeypatch.setattr(extract, "prepare_features", lambda *a: data)
    monkeypatch.setattr(extract, "load_backbone", lambda *a: Backbone())
    monkeypatch.setattr(extract, "FrozenASTPrefix", Prefix)
    extract.extract_ast(rows, tmp_path, tmp_path, tmp_path, torch.device("cpu"))
    assert np.load(tmp_path / "block10_tokens.npy").shape == (3, 146, 768)
    with np.load(tmp_path / "ast_embeddings.npz") as cache:
        assert cache["embeddings"].shape == (2, 768)
        np.testing.assert_array_equal(cache["embeddings"], 1)
    with np.load(tmp_path / "block10_index.npz") as cache:
        assert cache["segment_counts"].tolist() == [2, 1]


def test_weight_checksum_rejects_changed_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract.verify_weights(ROOT, "ast", tmp_path)
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        extract.verify_weights(ROOT, "ast", tmp_path)


def test_source_verifier_with_separate_audio_directory(tmp_path, monkeypatch):
    provenance = json.loads((ROOT / "data/provenance.json").read_text())
    manifest = pd.read_csv(ROOT / "data/manifest.csv")
    expected = dict(zip(manifest.filename, manifest.sha256))
    original_sha256 = prepare.sha256
    monkeypatch.setattr(prepare.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=provenance["revision"]))
    monkeypatch.setattr(prepare, "sha256", lambda p: expected[Path(p).name]
                        if Path(p).suffix == ".wav" else original_sha256(p))
    result = prepare.verify_source(ROOT, tmp_path, author_features=False, audio_dir=tmp_path / "audio")
    assert result["audio_files"] == 793 and result["analysis_calls"] == 792
    expected[manifest.iloc[0].filename] = "changed"
    with pytest.raises(ValueError, match="Audio differs"):
        prepare.verify_source(ROOT, tmp_path, author_features=False, audio_dir=tmp_path / "audio")
