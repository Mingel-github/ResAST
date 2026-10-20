import numpy as np
import pandas as pd
import pytest

from resast.evaluation.metrics import (
    PROBABILITIES, collect_oof, ensemble, metrics, summarize, units_to_cats, validate_predictions,
)


def predictions():
    return pd.DataFrame({
        "cat_id": ["001A", "002A", "003A"], "true_label": [0, 1, 2],
        "prob_kitten": [.8, .1, .1], "prob_adult": [.1, .8, .7], "prob_senior": [.1, .1, .2],
    })


def test_metrics_hand_calculation():
    score = metrics(predictions())
    assert score["accuracy"] == pytest.approx(2 / 3)
    assert score["macro_f1"] == pytest.approx((1 + 2 / 3) / 3)
    assert score["cross_entropy"] == pytest.approx(-np.log([.8, .8, .2]).mean())
    assert score["brier"] == pytest.approx((.06 + .06 + 1.14) / 3)
    assert score["recall_senior"] == 0


@pytest.mark.parametrize("case", ["duplicate", "invalid_label", "negative", "sum", "saved_label"])
def test_invalid_predictions(case):
    frame = predictions()
    if case == "duplicate":
        frame.loc[1, "cat_id"] = "001A"
    elif case == "invalid_label":
        frame.loc[1, "true_label"] = .5
    elif case == "negative":
        frame.loc[1, PROBABILITIES[0]] = -.1
    elif case == "sum":
        frame.loc[1, PROBABILITIES[0]] = .4
    else:
        frame["predicted_label"] = [0, 1, 2]
    with pytest.raises(ValueError):
        validate_predictions(frame)


def test_probability_ensemble_alignment_and_count():
    frame = predictions()
    combined = ensemble({17: frame, 43: frame.iloc[::-1], 101: frame})
    np.testing.assert_allclose(combined[list(PROBABILITIES)], frame[list(PROBABILITIES)], atol=1e-16)
    with pytest.raises(ValueError):
        ensemble({17: frame, 43: frame})
    changed = frame.copy()
    changed.loc[0, "true_label"] = 1
    with pytest.raises(ValueError):
        ensemble({17: frame, 43: changed, 101: frame})


def test_unit_mean_and_sample_sd():
    frame = predictions()
    units = pd.concat([frame, frame], ignore_index=True)
    cats = units_to_cats(units)
    np.testing.assert_array_equal(cats.unit_count, [2, 2, 2])
    assert metrics(cats) == metrics(frame)
    a, b = metrics(frame), metrics(frame)
    b["accuracy"] = 1
    result = summarize([a, b])
    assert result["accuracy"]["sd"] == pytest.approx(np.std([2 / 3, 1], ddof=1))


def test_oof_requires_exact_test_ids():
    frame = predictions()
    fourth = frame.iloc[[0]].assign(cat_id="004A")
    folds = {i: frame.iloc[[i]] for i in range(3)}
    folds[3] = fourth
    roles = pd.DataFrame([
        {"repeat": 0, "outer_fold": f, "role": "test", "cat_id": row.cat_id,
         "age_group": ("kitten", "adult", "senior")[row.true_label]}
        for f, data in folds.items() for row in data.itertuples()
    ])
    assert len(collect_oof(folds, roles, 0)) == 4
    with pytest.raises(ValueError):
        collect_oof({0: frame}, roles, 0)
    bad = {**folds, 3: frame.iloc[[0]]}
    with pytest.raises(ValueError):
        collect_oof(bad, roles, 0)
