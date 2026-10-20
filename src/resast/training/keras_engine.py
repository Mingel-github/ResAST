"""Author-feature baselines, preserving TensorFlow initialization and row batching."""

import math

import numpy as np
import pandas as pd

from resast.evaluation.metrics import PROBABILITIES, metrics, units_to_cats
from .common import TrainingConfig, check_fit, class_weights, standardizer
from .torch_engine import FitResult

SPECS = {
    "vggish129": {
        "dimensions": 129, "hidden": [128], "dropout": [0.44571035356880917],
        "optimizer": "Adamax", "learning_rate": 0.003109800273709165, "batch_size": 128,
    },
    "perch1281": {
        "dimensions": 1281, "hidden": [512, 64, 128, 32],
        "dropout": [0.15688606637942396, 0.28869063762474007, 0.27322054853254163, 0.4064145271744943],
        "optimizer": "Adam", "learning_rate": 0.007402414884293732, "batch_size": 32,
    },
    "yamnet": {
        "dimensions": 1024, "hidden": [480], "dropout": [0.40037235360372386],
        "optimizer": "Adamax", "learning_rate": 0.009272615392034748, "batch_size": 32,
    },
}


def default_config(name):
    spec = SPECS[name]
    return TrainingConfig(
        learning_rate=spec["learning_rate"], dropout=spec["dropout"][0],
        batch_size=spec["batch_size"], maximum_epochs=500, patience=30,
    )


def tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("Author-feature baselines require the TensorFlow 2.15 environment") from error
    tf.config.experimental.enable_op_determinism()
    return tf


def build_model(name, seed, config):
    tf = tensorflow()
    spec = SPECS[name]

    def build(dimensions):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(int(seed))
        layers = [tf.keras.layers.Input(shape=(dimensions,))]
        for i, (width, dropout) in enumerate(zip(spec["hidden"], spec["dropout"])):
            suffix = "" if name == "vggish129" else f"_{i}"
            layers.extend([
                tf.keras.layers.Dense(width, activation="relu", name=f"hidden{suffix}"),
                tf.keras.layers.BatchNormalization(name=f"batch_norm{suffix}"),
                tf.keras.layers.Dropout(dropout, seed=int(seed) + i, name=f"dropout{suffix}"),
            ])
        layers.append(tf.keras.layers.Dense(3, activation="softmax", name="output"))
        model = tf.keras.Sequential(layers)
        optimizer = getattr(tf.keras.optimizers, spec["optimizer"])
        model.compile(
            optimizer=optimizer(learning_rate=config.learning_rate, epsilon=config.optimizer_epsilon),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        )
        return model

    if name == "vggish129":
        reference = build(128)
        weights = [w.copy() for w in reference.get_weights()]
        model = build(129)
        kernel = np.zeros((129, 128), dtype=weights[0].dtype)
        kernel[:128] = weights[0]
        model.set_weights([kernel, *weights[1:]])
        return model
    return build(spec["dimensions"])


def predict(model, store, indices, config, preprocessing):
    x = ((store.frozen_embeddings[indices] - preprocessing["mean"]) / preprocessing["scale"]).astype(np.float32)
    p = model.predict(x, batch_size=config.batch_size, verbose=0)
    units = pd.DataFrame({
        "unit_id": store.call_ids[indices], "cat_id": store.cat_ids[indices],
        "true_label": store.labels[indices], **{name: p[:, i] for i, name in enumerate(PROBABILITIES)},
    }).sort_values("unit_id").reset_index(drop=True)
    return units_to_cats(units), units


def fit(name, store, train, config, seed, *, validation=None, fixed_epochs=None, on_epoch=None):
    check_fit(store, train, validation, fixed_epochs, config)
    mean, scale = standardizer(store.frozen_embeddings[train])
    preprocessing = {"mean": mean, "scale": scale}
    x = ((store.frozen_embeddings[train] - mean) / scale).astype(np.float32)
    y = store.labels[train].astype(np.int64)
    weights = class_weights(y)
    model = build_model(name, seed, config)
    training_seed = seed + 1_000_000
    tensorflow().keras.utils.set_random_seed(training_seed)
    best_loss, best_epoch, stale = math.inf, 0, 0
    best_state, best_cats, best_units, history = None, None, None, []
    maximum = config.maximum_epochs if fixed_epochs is None else fixed_epochs
    for epoch in range(1, maximum + 1):
        order = np.random.default_rng(training_seed + epoch).permutation(len(train))
        losses, sizes = [], []
        for start in range(0, len(order), config.batch_size):
            batch = order[start:start + config.batch_size]
            result = model.train_on_batch(x[batch], y[batch], sample_weight=weights[y[batch]], return_dict=True)
            loss = float(result["loss"] if isinstance(result, dict) else result)
            if not math.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            losses.append(loss)
            sizes.append(len(batch))
        row = {"epoch": epoch, "training_loss": float(np.average(losses, weights=sizes))}
        if validation is not None:
            cats, units = predict(model, store, validation, config, preprocessing)
            row["validation_metrics"] = metrics(cats)
            value = row["validation_metrics"]["cross_entropy"]
            if value < best_loss - config.minimum_delta:
                best_loss, best_epoch, stale = value, epoch, 0
                best_state = [w.copy() for w in model.get_weights()]
                best_cats, best_units = cats.copy(), units.copy()
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
        model.set_weights(best_state)
    else:
        best_epoch = maximum
    return FitResult(model, best_epoch, history, best_cats, best_units), preprocessing
