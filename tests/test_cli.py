import json
from types import SimpleNamespace

import pytest
import numpy as np
import pandas as pd

from resast.cli.train import cell_config, completed, locked_epochs
from resast.cli import evaluate, ensemble, train, lock, predict
from resast.training.common import sha256, write_json
from test_evaluation import predictions


@pytest.mark.parametrize("entry", [train.main, lock.main, evaluate.main, ensemble.main, predict.main])
def test_help(entry):
    with pytest.raises(SystemExit) as error:
        entry(["--help"])
    assert error.value.code == 0


def test_locked_configuration_selection():
    config = {"models": {"resast": {"cells": [
        {"repeat": 0, "fold": 0, "training": {"learning_rate": .012, "dropout": .25, "weight_decay": .001}},
    ]}}}
    assert cell_config(config, "resast", 0, 0).learning_rate == .012
    with pytest.raises(ValueError):
        cell_config(config, "resast", 0, 1)
    with pytest.raises(ValueError):
        cell_config({"models": {"ast": {}}}, "ast", 0, 0)
    with pytest.raises(ValueError):
        cell_config({"models": {"vggish129": {"training": {"learning_rate": .01}}}}, "vggish129", 0, 0)


def test_lock_explicit_small_scope(tmp_path):
    directory = train.run_directory(tmp_path, "select", "ast", 0, 0, 17)
    write_json(directory / "request.json", {
        "model": "ast", "repeat": 0, "fold": 0, "base_seed": 17, "stage": "select"})
    write_json(directory / "summary.json", {
        "status": "complete", "test_prediction_calls": 0, "best_epoch": 19,
        "artifacts": {"request.json": sha256(directory / "request.json")}})
    path = tmp_path / "epochs.json"
    args = ["--output", str(tmp_path), "--models", "ast", "--lock", str(path),
            "--repeats", "0", "--folds", "0", "--seeds", "17"]
    assert lock.main(args) is None
    result = json.loads(path.read_text())
    assert result["seeds"] == [17] and result["folds"] == [0]
    assert list(result["entries"]) == ["ast/0/0/17"]
    with pytest.raises(ValueError, match="Repeated"):
        lock.main([*args, "--seeds", "17", "17"])


def test_resume_validates_config_and_saved_content(tmp_path):
    directory = tmp_path / "run"
    request = {"model": "resast", "training": {"learning_rate": .006}}
    assert completed(directory, request, False) is None
    with pytest.raises(FileExistsError):
        completed(directory, request, False)
    assert completed(directory, request, True) is None
    artifact = directory / "history.json"
    write_json(artifact, [{"epoch": 1}])
    summary = {"status": "complete", "artifacts": {"history.json": sha256(artifact)}}
    write_json(directory / "summary.json", summary)
    assert completed(directory, request, True) == summary
    with pytest.raises(ValueError):
        completed(directory, {"model": "ast"}, True)
    write_json(artifact, [{"epoch": 2}])
    with pytest.raises(ValueError):
        completed(directory, request, True)


def test_refit_requires_matching_validation_lock(tmp_path):
    request = {"model": "resast", "repeat": 0, "fold": 0, "base_seed": 17, "stage": "select"}
    selection = tmp_path / "selection"
    write_json(selection / "request.json", request)
    summary = {"status": "complete", "best_epoch": 2, "artifacts": {"request.json": sha256(selection / "request.json")}}
    write_json(selection / "summary.json", summary)
    lock_path = tmp_path / "lock.json"
    write_json(lock_path, {"entries": {"resast/0/0/17": {"epoch": 2, "summary_sha256": sha256(selection / "summary.json")}}})
    refit = {**request, "stage": "refit", "selection_lock_sha256": sha256(lock_path)}
    assert locked_epochs(lock_path, selection, refit) == 2
    with pytest.raises(ValueError):
        locked_epochs(lock_path, selection, {**refit, "changed": True})


def test_evaluate_cli(tmp_path):
    path = tmp_path / "predictions.csv"
    predictions().to_csv(path, index=False)
    assert evaluate.main(["--predictions", str(path), "--output", str(tmp_path / "eval")]) is None
    result = json.loads((tmp_path / "eval/metrics.json").read_text())
    assert result["scores"][0]["n"] == 3


def test_training_cli_select_refit_resume(tmp_path, monkeypatch):
    from resast.data.features import FeatureStore
    rng = np.random.default_rng(1)
    store = FeatureStore(
        rng.normal(size=(24, 768)).astype(np.float32), np.array([f"call{i}" for i in range(24)]),
        np.repeat([f"cat{i}" for i in range(12)], 2), np.repeat(np.arange(12) % 3, 2).astype(np.int64),
    )
    roles = pd.DataFrame([
        {"cat_id": f"cat{i}", "repeat": 0, "outer_fold": 0,
         "role": "train" if i < 6 else "validation" if i < 9 else "test"}
        for i in range(12)
    ])
    monkeypatch.setattr(train, "load_inputs", lambda *a: (store, roles, None, None, {"test": "synthetic"}))
    config = tmp_path / "config.json"
    write_json(config, {"models": {"aves": {"training": {"maximum_epochs": 1, "patience": 1}}}})
    args = SimpleNamespace(config=config, model="aves", repeat=0, fold=0, seed=17,
        stage="select", device="cpu", output=tmp_path / "runs", selection_lock=None, resume=False)
    selected = train.run(args)
    assert selected["test_prediction_calls"] == 0
    directory = train.run_directory(args.output, "select", "aves", 0, 0, 17)
    args.selection_lock = tmp_path / "epochs.json"
    write_json(args.selection_lock, {"entries": {"aves/0/0/17": {
        "epoch": 1, "summary_sha256": sha256(directory / "summary.json"),
    }}})
    args.stage = "refit"
    refit = train.run(args)
    assert refit["test_prediction_calls"] == 1
    assert len(refit["train_ids"]) == 9 and len(refit["evaluated_ids"]) == 3
    args.resume = True
    assert train.run(args) == refit


def test_complete_oof_and_ensemble_commands(tmp_path):
    base = predictions()
    folds = [base.iloc[[i]] for i in range(3)] + [base.iloc[[0]].assign(cat_id="004A")]
    roles = pd.DataFrame([
        {"repeat": 0, "outer_fold": f, "role": "test", "cat_id": row.cat_id,
         "age_group": ("kitten", "adult", "senior")[row.true_label]}
        for f, frame in enumerate(folds) for row in frame.itertuples()
    ])
    role_path = tmp_path / "roles.csv"
    roles.to_csv(role_path, index=False)
    paths = []
    for i, frame in enumerate(folds):
        path = tmp_path / f"fold{i}.csv"
        frame.to_csv(path, index=False)
        paths.append(str(path))
    members = {}
    for seed in (17, 43, 101):
        destination = tmp_path / f"s{seed}"
        assert evaluate.main([
            "--predictions", *paths, "--roles", str(role_path), "--repeat", "0",
            "--model", "ast", "--seed", str(seed), "--output", str(destination),
        ]) is None
        members[seed] = destination / "oof_predictions.csv"
    args = [
        "--seed17", str(members[17]), "--seed43", str(members[43]), "--seed101", str(members[101]),
        "--roles", str(role_path), "--repeat", "0", "--output", str(tmp_path / "ensemble"),
    ]
    assert ensemble.main(args) is None
    record = json.loads((tmp_path / "ensemble/ensemble.json").read_text())
    assert record["metrics"]["n"] == 4
    provenance = json.loads((members[43].parent / "oof.json").read_text())
    write_json(members[43].parent / "oof.json", {**provenance, "repeat": 1})
    with pytest.raises(ValueError, match="repeat or seed"):
        ensemble.main(args)
