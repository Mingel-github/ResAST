import numpy as np
import pytest

from resast.training import keras_engine
from resast.evaluation.metrics import PROBABILITIES
from test_training import feature_store


@pytest.mark.parametrize("name", list(keras_engine.SPECS))
def test_author_baseline_short_fit_and_weights(name):
    pytest.importorskip("tensorflow")
    from dataclasses import replace
    cfg = replace(keras_engine.default_config(name), maximum_epochs=1, patience=1)
    store = feature_store(keras_engine.SPECS[name]["dimensions"])
    result, preprocessing = keras_engine.fit(
        name, store, np.arange(12), cfg, 17, validation=np.arange(12, 18),
    )
    assert result.best_epoch == 1
    restored = keras_engine.build_model(name, 17, cfg)
    restored.set_weights(result.model.get_weights())
    cats, _ = keras_engine.predict(restored, store, np.arange(12, 18), cfg, preprocessing)
    np.testing.assert_array_equal(cats[list(PROBABILITIES)], result.cats[list(PROBABILITIES)])
    np.testing.assert_array_equal(preprocessing["mean"], store.frozen_embeddings[:12].mean(0))


def test_vgg_frequency_dimension_starts_at_zero():
    pytest.importorskip("tensorflow")
    model = keras_engine.build_model("vggish129", 17, keras_engine.default_config("vggish129"))
    assert model.get_weights()[0].shape == (129, 128)
    np.testing.assert_array_equal(model.get_weights()[0][-1], np.zeros(128))
