"""Metrics for aligned, three-class identity predictions."""

import numpy as np
import pandas as pd

CLASSES = ("kitten", "adult", "senior")
PROBABILITIES = tuple(f"prob_{name}" for name in CLASSES)
METRICS = (
    "macro_f1", "accuracy", "balanced_accuracy", "recall_kitten",
    "recall_adult", "recall_senior", "cross_entropy", "brier",
)


def validate_predictions(frame: pd.DataFrame, *, unique_ids: bool = True) -> None:
    required = {"cat_id", "true_label", *PROBABILITIES}
    if not required <= set(frame) or frame.empty or frame["cat_id"].isna().any():
        raise ValueError("Expected nonempty identity, label and probability columns")
    if unique_ids and frame["cat_id"].duplicated().any():
        raise ValueError("Repeated identity in an identity-level prediction set")
    labels = frame["true_label"].to_numpy()
    if not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("Labels must use kitten=0, adult=1, senior=2")
    probabilities = frame[list(PROBABILITIES)].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Class probabilities must sum to one")
    if "predicted_label" in frame and not np.array_equal(frame["predicted_label"], probabilities.argmax(1)):
        raise ValueError("Saved predictions disagree with probability argmax")


def read_predictions(path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"cat_id": str, "unit_id": str}, float_precision="round_trip")
    validate_predictions(frame)
    return frame.sort_values("cat_id").reset_index(drop=True)


def units_to_cats(units: pd.DataFrame) -> pd.DataFrame:
    validate_predictions(units, unique_ids=False)
    if "unit_id" in units and units["unit_id"].duplicated().any():
        raise ValueError("Repeated feature unit")
    rows = []
    for cat_id, group in units.groupby("cat_id", sort=True):
        labels = group["true_label"].unique()
        if len(labels) != 1:
            raise ValueError(f"Inconsistent labels for {cat_id}")
        # Average unit probabilities per ID before taking argmax.
        p = group[list(PROBABILITIES)].to_numpy(dtype=np.float64).mean(axis=0)
        rows.append({
            "cat_id": str(cat_id), "true_label": int(labels[0]), "unit_count": len(group),
            **dict(zip(PROBABILITIES, p)), "predicted_label": int(p.argmax()),
        })
    return pd.DataFrame(rows).sort_values("cat_id").reset_index(drop=True)


def metrics(frame: pd.DataFrame) -> dict:
    validate_predictions(frame)
    ordered = frame.sort_values("cat_id")
    y = ordered["true_label"].to_numpy(dtype=np.int64)
    p = ordered[list(PROBABILITIES)].to_numpy(dtype=np.float64)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (y, p.argmax(axis=1)), 1)
    tp = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1)
    denominator = support + confusion.sum(axis=0)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(3), where=denominator != 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support != 0)
    return {
        "n": len(y), "macro_f1": float(f1.mean()), "accuracy": float(tp.sum() / len(y)),
        "balanced_accuracy": float(recall.mean()),
        **{f"recall_{name}": float(recall[i]) for i, name in enumerate(CLASSES)},
        "cross_entropy": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "brier": float(np.square(p - np.eye(3)[y]).sum(axis=1).mean()),
    }


def collect_oof(folds: dict[int, pd.DataFrame], roles: pd.DataFrame, repeat: int) -> pd.DataFrame:
    if set(folds) != {0, 1, 2, 3}:
        raise ValueError("Complete OOF evaluation requires all four folds")
    for fold, frame in folds.items():
        validate_predictions(frame)
        expected = roles[(roles.repeat == repeat) & (roles.outer_fold == fold) & (roles.role == "test")]
        if expected.cat_id.duplicated().any() or set(frame.cat_id) != set(expected.cat_id):
            raise ValueError(f"Fold {fold} differs from the held-out identity set")
        if "age_group" in expected:
            labels = expected.set_index("cat_id")["age_group"].map(dict(zip(CLASSES, range(3))))
            if labels.isna().any() or not np.array_equal(frame.true_label, labels.loc[frame.cat_id]):
                raise ValueError("Prediction labels differ from the fixed role table")
    combined = pd.concat([folds[f] for f in range(4)], ignore_index=True)
    validate_predictions(combined)
    return combined.sort_values("cat_id").reset_index(drop=True)


def ensemble(frames: dict[int, pd.DataFrame], seeds=(17, 43, 101)) -> pd.DataFrame:
    if len(set(seeds)) != len(seeds) or set(frames) != set(seeds):
        raise ValueError("Exactly the declared distinct seed members are required")
    aligned = []
    for seed in seeds:
        validate_predictions(frames[seed])
        aligned.append(frames[seed].sort_values("cat_id").reset_index(drop=True))
    first = aligned[0]
    for frame in aligned[1:]:
        if not first[["cat_id", "true_label"]].equals(frame[["cat_id", "true_label"]]):
            raise ValueError("Ensemble members must share identities and labels")
    combined = first[["cat_id", "true_label"]].copy()
    # Give each aligned seed equal weight in probability space.
    p = np.stack([f[list(PROBABILITIES)].to_numpy(dtype=np.float64) for f in aligned]).mean(axis=0)
    for i, name in enumerate(PROBABILITIES):
        combined[name] = p[:, i]
    combined["predicted_label"] = p.argmax(axis=1)
    return combined


def summarize(scores: list[dict]) -> dict:
    if len(scores) < 2:
        raise ValueError("Sample standard deviation needs at least two scores")
    return {
        key: {"mean": float(np.mean([s[key] for s in scores])),
              "sd": float(np.std([s[key] for s in scores], ddof=1)), "n": len(scores)}
        for key in METRICS
    }
