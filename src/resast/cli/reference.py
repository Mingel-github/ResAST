"""Verify and re-score the released prediction bundle without training."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from resast.evaluation.metrics import CLASSES, METRICS, PROBABILITIES, ensemble, metrics, read_predictions, summarize
from resast.training.common import sha256, write_json


def verify(root):
    root = Path(root)
    index = json.loads((root / "results/reference/index.json").read_text(encoding="utf-8"))
    if sha256(root / "splits/roles.csv") != index["roles_sha256"]:
        raise ValueError("Role table checksum differs")
    roles = pd.read_csv(root / "splits/roles.csv", dtype={"cat_id": str})
    labels = roles[["cat_id", "age_group"]].drop_duplicates().set_index("cat_id").age_group
    labels = labels.map(dict(zip(CLASSES, range(3)))).sort_index()
    result, files = [], 0
    for record in index["models"]:
        cached = {}
        for mode in ("single", "ensemble"):
            expected = ({(r, s) for r in index["repeats"] for s in index["seeds"]}
                        if mode == "single" else {(r, None) for r in index["repeats"]})
            keys = [(e["repeat"], e.get("seed")) for e in record[mode]]
            if len(keys) != len(expected) or set(keys) != expected:
                raise ValueError("Incomplete or duplicate prediction entries")
            scores = []
            for entry in record[mode]:
                path = (root / entry["path"]).resolve()
                if not path.is_relative_to(root.resolve()) or sha256(path) != entry["sha256"]:
                    raise ValueError("Prediction path or checksum differs")
                frame = read_predictions(path)
                if list(frame.cat_id) != list(labels.index) or not np.array_equal(frame.true_label, labels):
                    raise ValueError("Predictions differ from the full labelled ID set")
                score = metrics(frame)
                for key in METRICS:
                    if abs(score[key] - entry["metrics"][key]) > 1e-12:
                        raise ValueError(f"Reference metric differs: {record['model']}/{key}")
                scores.append(score)
                if mode == "single":
                    cached[(entry["repeat"], entry["seed"])] = frame
                else:
                    averaged = ensemble({s: cached[entry["repeat"], s] for s in index["seeds"]})
                    if not np.allclose(averaged[list(PROBABILITIES)], frame[list(PROBABILITIES)],
                                       atol=1e-12, rtol=0):
                        raise ValueError("Saved ensemble differs from equal probability averaging")
                files += 1
            summary = summarize(scores)
            for metric, stats in summary.items():
                target = record[f"{mode}_summary"][metric]
                if abs(stats["mean"] - target["mean"]) > 1e-12 or abs(stats["sd"] - target["sample_sd"]) > 1e-12:
                    raise ValueError("Reference summary differs")
                result.append({"model": record["model"], "mode": mode, "metric": metric, **stats})
    stored = pd.read_csv(root / "results/summary.csv", float_precision="round_trip")
    actual = pd.DataFrame(result)
    merged = stored.merge(actual, on=["model", "mode", "metric"], validate="one_to_one",
                          suffixes=("_stored", "_actual"))
    if len(merged) != len(actual) or len(stored) != len(actual):
        raise ValueError("Incomplete summary table")
    if not np.allclose(merged.mean_stored, merged.mean_actual, atol=1e-12, rtol=0):
        raise ValueError("Stored table means differ")
    if not np.allclose(merged.sample_sd, merged.sd, atol=1e-12, rtol=0):
        raise ValueError("Stored table standard deviations differ")
    return {"status": "verified", "prediction_files": files, "models": len(index["models"]),
            "rows": result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = verify(args.root)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, result)
    print(f"Verified {result['prediction_files']} predictions for {result['models']} models.")
    print("Model                 Single F1       Ensemble F1")
    for model in dict.fromkeys(r["model"] for r in result["rows"]):
        values = [r for r in result["rows"] if r["model"] == model and r["metric"] == "macro_f1"]
        print(f"{model:21s} " + "    ".join(f"{100*v['mean']:.2f} +/- {100*v['sd']:.2f}" for v in values))


if __name__ == "__main__":
    main()
