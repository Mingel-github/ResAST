"""Optionally repeat the validation-only AST grid search for one model and split."""

import argparse
from dataclasses import asdict
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch

from resast.cli.train import cell_config, load_inputs, read_json, source_code_digest
from resast.data.dataset import role_indices
from resast.evaluation.metrics import metrics
from resast.models.classifier import MODEL_FLAGS
from resast.training.common import TrainingConfig, configure_determinism, full_seed, sha256, write_json
from resast.training.torch_engine import fit


def choose(candidates, tolerance=1e-12):
    pool = list(candidates)
    if not pool:
        raise ValueError("Search candidates are empty")
    for key, maximize in (("mean_macro_f1", True), ("mean_accuracy", True), ("mean_brier", False)):
        values = [c[key] for c in pool]
        if not np.isfinite(values).all():
            raise ValueError("Search scores must be finite")
        best = max(values) if maximize else min(values)
        pool = [c for c in pool if abs(c[key] - best) <= tolerance]
    return min(pool, key=lambda c: c["configuration_id"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--grid", type=Path, default=Path("configs/search.json"))
    parser.add_argument("--model", required=True, choices=tuple(MODEL_FLAGS))
    parser.add_argument("--repeat", required=True, type=int, choices=range(3))
    parser.add_argument("--fold", required=True, type=int, choices=range(4))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh search output directory")
    configure_determinism()
    document, grid = read_json(args.config), read_json(args.grid)
    store, roles, acoustic, backbone, hashes = load_inputs(document, args.config.resolve().parent, args.model)
    indices = role_indices(store, roles, args.repeat, args.fold)
    base = asdict(cell_config(document, args.model, args.repeat, args.fold))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "request.json", {
        "model": args.model, "repeat": args.repeat, "fold": args.fold, "inputs": hashes,
        "grid_sha256": sha256(args.grid), "source_code_sha256": source_code_digest(),
        "device": args.device, "torch": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
    })
    candidates = []
    for candidate in grid["grid"]:
        overrides = {k: v for k, v in candidate.items() if k != "configuration_id"}
        cfg = TrainingConfig(**{**base, **overrides})
        scores = []
        for seed in grid["seeds"]:
            result = fit(args.model, store, indices["train"], cfg, torch.device(args.device),
                         full_seed(seed, args.repeat, args.fold), validation=indices["validation"],
                         backbone=backbone, acoustic=acoustic)
            score = metrics(result.cats)
            scores.append(score)
            write_json(args.output / f"{candidate['configuration_id']}_seed_{seed}.json", {
                "configuration": candidate, "base_seed": seed, "best_epoch": result.best_epoch,
                "metrics": score, "history": result.history, "test_prediction_calls": 0,
            })
            del result
            print(f"{candidate['configuration_id']} seed={seed} complete", flush=True)
        candidates.append({
            "configuration_id": candidate["configuration_id"],
            **{f"mean_{key}": float(np.mean([s[key] for s in scores]))
               for key in ("macro_f1", "accuracy", "brier")},
        })
    selected = choose(candidates, grid["tie_tolerance"])
    write_json(args.output / "selection.json", {"selected": selected, "candidates": candidates,
                                              "test_prediction_calls": 0})
    print(f"Selected {selected['configuration_id']} on validation IDs.")


if __name__ == "__main__":
    main()
