import numpy as np
import pytest
import torch

from resast.data.dataset import TokenStore
from resast.data.features import FeatureStore
from resast.evaluation.metrics import PROBABILITIES
from resast.models.classifier import MODEL_FLAGS
from resast.training import torch_engine
from resast.training.checkpoint import save_checkpoint, load_checkpoint
from resast.training.common import TrainingConfig, check_fit, class_weights, full_seed


def feature_store(dimensions=768):
    rng = np.random.default_rng(20261020)
    n = 18
    return FeatureStore(
        rng.normal(size=(n, dimensions)).astype(np.float32),
        np.asarray([f"call{i}" for i in range(n)]),
        np.asarray([f"cat{i // 2:02d}" for i in range(n)]),
        np.repeat(np.arange(9) % 3, 2).astype(np.int64),
    )


def token_store():
    source = feature_store()
    rng = np.random.default_rng(77)
    return TokenStore(
        rng.normal(size=(18, 146, 768)).astype(np.float32),
        tuple(np.asarray([i], dtype=np.int64) for i in range(18)),
        source.frozen_embeddings, source.call_ids, source.cat_ids, source.labels,
    )


def test_seed_weights_and_fit_roles():
    assert full_seed(43, 2, 3) == 20343
    np.testing.assert_array_equal(class_weights(np.array([0, 0, 1, 2])), np.array([2/3, 4/3, 4/3], dtype=np.float32))
    with pytest.raises(ValueError):
        class_weights(np.array([0, 1]))
    store = feature_store()
    with pytest.raises(ValueError, match="overlap"):
        check_fit(store, np.arange(0, 12, 2), np.arange(1, 12, 2), None, TrainingConfig())
    with pytest.raises(ValueError):
        check_fit(store, np.arange(12), None, 0, TrainingConfig())


@pytest.mark.parametrize("name", list(MODEL_FLAGS))
def test_ast_training_and_checkpoint_roundtrip(backbone, tmp_path, name):
    store = token_store()
    acoustic = np.random.default_rng(33).normal(size=(18, 20)).astype(np.float32)
    config = TrainingConfig(maximum_epochs=1, patience=1)
    result = torch_engine.fit(
        name, store, np.arange(12), config, torch.device("cpu"), 17,
        validation=np.arange(12, 18), backbone=backbone, acoustic=acoustic,
    )
    assert result.best_epoch == 1 and len(result.cats) == 3
    assert all(p.grad is None for p in result.model.tail.block11.parameters())
    path = tmp_path / "model.pt"
    save_checkpoint(path, result.model, name, config, 17, result.best_epoch)
    loaded, _ = load_checkpoint(path, torch.device("cpu"), backbone=backbone)
    cats, _ = torch_engine.predict(loaded, store, np.arange(12, 18), config, torch.device("cpu"), 17, acoustic)
    np.testing.assert_array_equal(cats[list(PROBABILITIES)], result.cats[list(PROBABILITIES)])


@pytest.mark.parametrize("name,dimensions", [("cnn14", 2048), ("aves", 768), ("ssast", 768)])
def test_frozen_head_selection_and_fresh_refit(name, dimensions, tmp_path):
    store = feature_store(dimensions)
    cfg = TrainingConfig(maximum_epochs=2, patience=1)
    device = torch.device("cpu")
    selection = torch_engine.fit(name, store, np.arange(12), cfg, device, 43, validation=np.arange(12, 18))
    refit = torch_engine.fit(name, store, np.arange(18), cfg, device, 43, fixed_epochs=selection.best_epoch)
    assert refit.cats is None
    np.testing.assert_array_equal(refit.model.embedding_mean.numpy(), store.frozen_embeddings.mean(0))
    np.testing.assert_array_equal(selection.model.embedding_mean.numpy(), store.frozen_embeddings[:12].mean(0))
    path = tmp_path / "model.pt"
    save_checkpoint(path, refit.model, name, cfg, 43, refit.best_epoch)
    restored, _ = load_checkpoint(path, device)
    for key, value in restored.state_dict().items():
        assert torch.equal(value, refit.model.state_dict()[key])


def test_fixed_seed_repeats_same_training():
    store = feature_store()
    cfg = TrainingConfig(maximum_epochs=1)
    kwargs = dict(name="aves", store=store, train=np.arange(12), config=cfg, device=torch.device("cpu"),
                  seed=101, fixed_epochs=1)
    first, second = torch_engine.fit(**kwargs), torch_engine.fit(**kwargs)
    for key, tensor in first.model.state_dict().items():
        assert torch.equal(tensor, second.model.state_dict()[key])


def test_minimum_delta_patience_restores_first_checkpoint(monkeypatch):
    store = feature_store()
    cfg = TrainingConfig(maximum_epochs=5, patience=2)
    from test_evaluation import predictions
    monkeypatch.setattr(torch_engine, "predict", lambda *args, **kwargs: (predictions(), predictions()))
    result = torch_engine.fit("aves", store, np.arange(12), cfg, torch.device("cpu"), 17, validation=np.arange(12, 18))
    assert result.best_epoch == 1
    assert len(result.history) == 3
