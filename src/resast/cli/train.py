"""Run one declared model/repeat/fold/seed, with fit-boundary recovery."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch

from resast.data.dataset import load_acoustic_features, load_roles, load_token_store, role_indices, validate_roles
from resast.data.features import load_features, load_author_features
from resast.evaluation.metrics import CLASSES, metrics
from resast.models.ast import load_backbone
from resast.models.classifier import MODEL_FLAGS
from resast.models.frozen_head import DIMENSIONS
from resast.training import keras_engine, torch_engine
from resast.training.checkpoint import save_checkpoint, load_checkpoint
from resast.training.common import TrainingConfig, configure_determinism, full_seed, sha256, write_json

MODELS = (*MODEL_FLAGS, *DIMENSIONS, *keras_engine.SPECS)
SEEDS = (17, 43, 101)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def cell_config(document, name, repeat, fold):
    base = keras_engine.default_config(name) if name in keras_engine.SPECS else TrainingConfig()
    spec = document["models"][name]
    overrides = spec.get("training", {})
    if "cells" in spec:
        matches = [x for x in spec["cells"] if x["repeat"] == repeat and x["fold"] == fold]
        if len(matches) != 1:
            raise ValueError("Exactly one locked configuration is required for this model/fold")
        overrides = {**overrides, **matches[0]["training"]}
    if name in MODEL_FLAGS and not overrides:
        raise ValueError("AST training requires an explicit recipe or locked cell configuration")
    config = TrainingConfig(**{**asdict(base), **overrides})
    if name in keras_engine.SPECS:
        # These public comparison pipelines retain their author-head recipes.
        for field in ("learning_rate", "dropout", "weight_decay", "optimizer_epsilon", "batch_size"):
            if getattr(config, field) != getattr(base, field):
                raise ValueError(f"The fixed author baseline recipe changed: {field}")
    return config


def load_inputs(document, config_dir, name):
    hashes = {}

    def resolve(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (config_dir / path).resolve()

    def source(key):
        value = document["data"][key]
        path = resolve(value["path"] if isinstance(value, dict) else value)
        digest = sha256(path)
        if isinstance(value, dict) and value.get("sha256", digest) != digest:
            raise ValueError(f"Source checksum mismatch: {key}")
        hashes[key] = digest
        return path

    roles = load_roles(source("roles"))
    acoustic, backbone = None, None
    if name in MODEL_FLAGS:
        store = load_token_store(source("tokens"), source("token_index"), source("fbank"), source("ast_embeddings"))
        if MODEL_FLAGS[name][1]:
            acoustic = load_acoustic_features(source("acoustic"), store.call_ids)
        # The release takes a downloaded, pinned snapshot rather than an arbitrary model.
        model_path = resolve(document["backbone"])
        if "backbone_checksums" in document:
            specification = read_json(resolve(document["backbone_checksums"]))["ast"]
            for filename, expected in specification["files"].items():
                if sha256(model_path / filename) != expected:
                    raise ValueError(f"Pretrained AST checksum differs: {filename}")
        for filename in ("config.json", "model.safetensors"):
            hashes[f"backbone/{filename}"] = sha256(model_path / filename)
        backbone = load_backbone(model_path)
    elif name in keras_engine.SPECS:
        store = load_author_features(source(name), name, roles.cat_id.unique())
    else:
        store = load_features(source(name), DIMENSIONS[name])
    validate_roles(store, roles)
    labels = roles[["cat_id", "age_group"]].drop_duplicates()
    if labels.cat_id.duplicated().any():
        raise ValueError("Role table has conflicting labels")
    lookup = labels.set_index("cat_id").age_group.map(dict(zip(CLASSES, range(3))))
    if lookup.isna().any() or not np.array_equal(lookup.loc[store.cat_ids].to_numpy(), store.labels):
        raise ValueError("Feature labels differ from the role bank")
    return store, roles, acoustic, backbone, hashes


def run_directory(root, stage, name, repeat, fold, seed):
    return Path(root) / stage / name / f"repeat_{repeat}" / f"fold_{fold}" / f"seed_{seed}"


def source_code_digest():
    import hashlib
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def completed(directory, request, resume):
    marker = directory / "request.json"
    if marker.exists():
        if read_json(marker) != request:
            raise ValueError("Run directory belongs to a different configuration, input or code version")
        if not resume:
            raise FileExistsError("Run exists; use --resume to validate and reuse completed work")
        summary_path = directory / "summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            if summary["status"] != "complete":
                raise ValueError("Unexpected completion status")
            for filename, digest in summary["artifacts"].items():
                if sha256(directory / filename) != digest:
                    raise ValueError(f"Saved artifact changed: {filename}")
            return summary
    elif directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Output directory contains unrelated files")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(marker, request)
    return None


def locked_epochs(lock_path, selection_dir, request):
    lock = read_json(lock_path)
    key = "/".join(str(request[k]) for k in ("model", "repeat", "fold", "base_seed"))
    entry = lock["entries"][key]
    if sha256(selection_dir / "summary.json") != entry["summary_sha256"]:
        raise ValueError("Selection summary differs from the epoch lock")
    expected = {**request, "stage": "select"}
    expected.pop("selection_lock_sha256", None)
    if read_json(selection_dir / "request.json") != expected:
        raise ValueError("Selection and refit settings differ")
    summary = read_json(selection_dir / "summary.json")
    for filename, digest in summary["artifacts"].items():
        if sha256(selection_dir / filename) != digest:
            raise ValueError("Selection artifact changed after locking")
    if summary["best_epoch"] != entry["epoch"]:
        raise ValueError("Selected epoch differs from its lock")
    return entry["epoch"]


def run(args):
    configure_determinism()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    document = read_json(args.config)
    config = cell_config(document, args.model, args.repeat, args.fold)
    store, roles, acoustic, backbone, hashes = load_inputs(document, Path(args.config).resolve().parent, args.model)
    indices = role_indices(store, roles, args.repeat, args.fold)
    seed = full_seed(args.seed, args.repeat, args.fold)
    request = {
        "model": args.model, "repeat": args.repeat, "fold": args.fold, "base_seed": args.seed,
        "full_seed": seed, "stage": args.stage, "training": asdict(config),
        "inputs": hashes, "source_code_sha256": source_code_digest(), "device": args.device,
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": str(np.__version__), "cpu_threads": torch.get_num_threads(),
    }
    if args.model in keras_engine.SPECS:
        request["tensorflow"] = keras_engine.tensorflow().__version__
    fixed_epochs = None
    if args.stage == "refit":
        if args.selection_lock is None:
            raise ValueError("Refitting requires a locked validation epoch table")
        request["selection_lock_sha256"] = sha256(args.selection_lock)
        selection_dir = run_directory(args.output, "select", args.model, args.repeat, args.fold, args.seed)
        fixed_epochs = locked_epochs(args.selection_lock, selection_dir, request)
        train = np.sort(np.concatenate([indices["train"], indices["validation"]]))
        validation = None
    else:
        train, validation = indices["train"], indices["validation"]
    directory = run_directory(args.output, args.stage, args.model, args.repeat, args.fold, args.seed)
    previous = completed(directory, request, args.resume)
    if previous is not None:
        print(f"Verified completed fit: {directory}", flush=True)
        return previous

    def log_epoch(row):
        print(f"{args.model} {args.stage} repeat={args.repeat} fold={args.fold} seed={args.seed} epoch={row['epoch']}", flush=True)

    if args.model in keras_engine.SPECS:
        result, preprocessing = keras_engine.fit(
            args.model, store, train, config, seed, validation=validation,
            fixed_epochs=fixed_epochs, on_epoch=log_epoch,
        )
        weights_path = directory / "model.npz"
        np.savez_compressed(
            weights_path, **preprocessing,
            **{f"weight_{i:03d}": w for i, w in enumerate(result.model.get_weights())},
        )
        restored = keras_engine.build_model(args.model, seed, config)
        with np.load(weights_path, allow_pickle=False) as data:
            restored.set_weights([data[key] for key in sorted(data.files) if key.startswith("weight_")])
            preprocessing = {key: data[key].copy() for key in ("mean", "scale")}
    else:
        result = torch_engine.fit(
            args.model, store, train, config, device, seed, validation=validation,
            fixed_epochs=fixed_epochs, backbone=backbone, acoustic=acoustic, on_epoch=log_epoch,
        )
        weights_path = directory / "model.pt"
        save_checkpoint(weights_path, result.model, args.model, config, seed, result.best_epoch)
        restored, _ = load_checkpoint(weights_path, device, backbone=backbone)
    write_json(directory / "history.json", result.history)
    # Persist the fixed checkpoint before making any held-out prediction.
    write_json(directory / "checkpoint_lock.json", {
        "model_sha256": sha256(weights_path), "epoch": result.best_epoch,
        "test_predictions_before_lock": 0, "stage": args.stage,
    })
    evaluation_indices = indices["test"] if args.stage == "refit" else indices["validation"]
    if args.model in keras_engine.SPECS:
        cats, units = keras_engine.predict(restored, store, evaluation_indices, config, preprocessing)
    else:
        cats, units = torch_engine.predict(restored, store, evaluation_indices, config, device, seed, acoustic)
    if result.cats is not None:
        from resast.evaluation.metrics import PROBABILITIES
        if not cats.cat_id.equals(result.cats.cat_id) or not np.allclose(
            cats[list(PROBABILITIES)], result.cats[list(PROBABILITIES)], atol=1e-6, rtol=0,
        ):
            raise RuntimeError("Checkpoint readback changed validation predictions")
    # Retain the historical author-baseline CSV precision.
    short_csv = args.model in ("perch1281", "yamnet")
    csv_options = {"float_format": "%.10g" if short_csv else "%.17g",
                   "lineterminator": None if short_csv else "\n"}
    cats.to_csv(directory / "cat_predictions.csv", index=False, **csv_options)
    units.to_csv(directory / "unit_predictions.csv", index=False, **csv_options)
    names = ("request.json", "history.json", "checkpoint_lock.json", weights_path.name,
             "cat_predictions.csv", "unit_predictions.csv")
    summary = {
        "status": "complete", "best_epoch": result.best_epoch, "metrics": metrics(cats),
        "train_ids": sorted(set(store.cat_ids[train])), "evaluated_ids": sorted(set(store.cat_ids[evaluation_indices])),
        "test_prediction_calls": int(args.stage == "refit"),
        "artifacts": {name: sha256(directory / name) for name in names},
    }
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary["metrics"], indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--stage", required=True, choices=("select", "refit"))
    parser.add_argument("--repeat", required=True, type=int, choices=range(3))
    parser.add_argument("--fold", required=True, type=int, choices=range(4))
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selection-lock", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
