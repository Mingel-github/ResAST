"""Lock validation-selected epochs after all declared fits are complete."""

import argparse
from pathlib import Path

from .train import MODELS, SEEDS, read_json, run_directory
from resast.training.common import sha256, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--repeats", nargs="+", type=int, choices=range(3), default=list(range(3)))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(4), default=list(range(4)))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    args = parser.parse_args(argv)
    for name in ("models", "repeats", "folds", "seeds"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            raise ValueError(f"Repeated {name}")
    entries = {}
    for name in args.models:
        for repeat in args.repeats:
            for fold in args.folds:
                for seed in args.seeds:
                    directory = run_directory(args.output, "select", name, repeat, fold, seed)
                    request = read_json(directory / "request.json")
                    expected = {"model": name, "repeat": repeat, "fold": fold, "base_seed": seed, "stage": "select"}
                    if any(request[k] != v for k, v in expected.items()):
                        raise ValueError("Selection coordinate mismatch")
                    summary = read_json(directory / "summary.json")
                    if summary["status"] != "complete" or summary["test_prediction_calls"] != 0:
                        raise ValueError("Expected completed validation-only selection")
                    for filename, digest in summary["artifacts"].items():
                        if sha256(directory / filename) != digest:
                            raise ValueError("Selection artifact changed")
                    entries[f"{name}/{repeat}/{fold}/{seed}"] = {
                        "epoch": summary["best_epoch"], "summary_sha256": sha256(directory / "summary.json"),
                    }
    lock = {"schema": 1, "models": args.models, "repeats": args.repeats,
            "folds": args.folds, "seeds": args.seeds, "entries": entries}
    if args.lock.exists() and read_json(args.lock) != lock:
        raise FileExistsError("A different epoch lock already exists")
    write_json(args.lock, lock)
    print(f"Locked {len(entries)} selected epoch counts: {args.lock}")


if __name__ == "__main__":
    main()
