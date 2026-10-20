"""Equal-probability ensemble of the three fixed seeds within one repeat."""

import argparse
import json
from pathlib import Path

from resast.data.dataset import load_roles
from resast.evaluation.metrics import CLASSES, ensemble, metrics, read_predictions
from resast.training.common import sha256, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed17", type=Path, required=True)
    parser.add_argument("--seed43", type=Path, required=True)
    parser.add_argument("--seed101", type=Path, required=True)
    parser.add_argument("--repeat", type=int, choices=range(3), required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {17: args.seed17, 43: args.seed43, 101: args.seed101}
    if len({p.resolve() for p in paths.values()}) != 3:
        raise ValueError("Provide three distinct member files")
    frames = {seed: read_predictions(path) for seed, path in paths.items()}
    models = set()
    for seed, path in paths.items():
        provenance = json.loads((path.parent / "oof.json").read_text(encoding="utf-8"))
        if provenance["seed"] != seed or provenance["repeat"] != args.repeat:
            raise ValueError("OOF member uses a different repeat or seed")
        if provenance["predictions_sha256"] != sha256(path) or provenance["roles_sha256"] != sha256(args.roles):
            raise ValueError("OOF member or role table changed")
        models.add(provenance["model"])
    if len(models) != 1:
        raise ValueError("Seed ensemble members must use the same model")
    roles = load_roles(args.roles)
    labels = roles[roles.repeat == args.repeat][["cat_id", "age_group"]].drop_duplicates().set_index("cat_id")
    if labels.index.duplicated().any():
        raise ValueError("Role labels conflict")
    lookup = labels.age_group.map(dict(zip(CLASSES, range(3))))
    for frame in frames.values():
        if set(frame.cat_id) != set(lookup.index) or not (frame.true_label.to_numpy() == lookup.loc[frame.cat_id].to_numpy()).all():
            raise ValueError("Expected a complete same-cohort OOF member")
    combined = ensemble(frames)
    args.output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output / "ensemble_predictions.csv", index=False, float_format="%.17g")
    result = {
        "model": next(iter(models)), "repeat": args.repeat, "seeds": [17, 43, 101],
        "rule": "arithmetic_mean_of_class_probabilities",
        "members": {str(s): {"path": str(p), "sha256": sha256(p)} for s, p in paths.items()},
        "metrics": metrics(combined),
    }
    write_json(args.output / "ensemble.json", result)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
