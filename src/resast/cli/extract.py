"""Build label-independent feature caches from verified local audio and weights."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch

from resast.cli.prepare import verify_source
from resast.data.acoustic import FEATURE_NAMES, extract_call_features
from resast.data.audio import average_frozen_segments, load_audio, make_feature_extractor, prepare_features
from resast.data.dataset import load_clean_manifest
from resast.models.ast import FrozenASTPrefix, load_backbone
from resast.training.common import configure_determinism, sha256, write_json


def verify_weights(root, name, directory):
    sources = json.loads((Path(root) / "configs/pretrained.json").read_text(encoding="utf-8"))
    spec = sources[name]
    for filename, expected in spec["files"].items():
        if sha256(Path(directory) / filename) != expected:
            raise ValueError(f"Pretrained file checksum differs: {name}/{filename}")
    return spec


def default_sdpa():
    """The original embedding extractors used the framework's default SDP routing."""
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def identities(rows):
    return {
        "call_ids": rows.filename.to_numpy(str),
        "cat_ids": rows.analysis_cat_id.to_numpy(str),
        "labels": rows.age_group_filename.map({"kitten": 0, "adult": 1, "senior": 2}).to_numpy(np.int64),
    }


def extract_ast(rows, upstream, weights, output, device):
    data = prepare_features(rows, upstream, make_feature_extractor(weights))
    np.savez_compressed(output / "ast_fbank.npz", **data)
    model = load_backbone(weights).to(device)
    default_sdpa()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(data["features"]), 8):
            batch = torch.from_numpy(data["features"][start:start + 8]).to(device)
            # Preserve the full original forward path used for standardizer embeddings.
            parts.append(model(input_values=batch).pooler_output.cpu().numpy().astype(np.float32))
    embeddings = average_frozen_segments(
        np.concatenate(parts), data["segment_call_indices"], data["segment_counts"],
    )
    np.savez_compressed(output / "ast_embeddings.npz", embeddings=embeddings, **identities(rows))
    configure_determinism()
    prefix = FrozenASTPrefix(model).to(device)
    tokens = np.lib.format.open_memmap(
        output / "block10_tokens.npy", mode="w+", dtype=np.float32,
        shape=(len(data["features"]), 146, 768),
    )
    for start in range(0, len(tokens), 8):
        batch = torch.from_numpy(data["features"][start:start + 8]).to(device)
        tokens[start:start + len(batch)] = prefix(batch).cpu().numpy()
        print(f"AST prefix {min(start + 8, len(tokens))}/{len(tokens)}", flush=True)
    tokens.flush()
    del tokens
    np.savez_compressed(
        output / "block10_index.npz",
        **{k: data[k] for k in ("segment_call_indices", "segment_counts", "call_ids", "source_paths")},
        block_output_one_based=np.asarray([10]), patch_token_start=np.asarray([2]),
    )


def torchaudio_waveform(path, rate, minimum=0):
    import torchaudio
    waveform, original_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if original_rate != rate:
        waveform = torchaudio.functional.resample(waveform, original_rate, rate)
    if waveform.numel() < minimum:
        waveform = torch.nn.functional.pad(waveform, (0, minimum - waveform.numel()))
    return waveform


def cnn14_class():
    # Import the architecture without panns_inference's unrelated label-CSV download.
    spec = importlib.util.find_spec("panns_inference")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("Install the feature-baselines extra")
    directory = Path(next(iter(spec.submodule_search_locations)))
    package = types.ModuleType("panns_inference")
    package.__path__ = [str(directory)]
    package.__package__ = "panns_inference"
    sys.modules["panns_inference"] = package
    module_spec = importlib.util.spec_from_file_location("panns_inference.models", directory / "models.py")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module.Cnn14


def extract_baseline(name, rows, upstream, weights, device):
    default_sdpa()
    if name == "ssast":
        from transformers import ASTModel, AutoFeatureExtractor
        extractor = AutoFeatureExtractor.from_pretrained(str(weights), local_files_only=True)
        model = ASTModel.from_pretrained(str(weights), use_safetensors=True, local_files_only=True).to(device).eval()
        parts = []
        for start in range(0, len(rows), 8):
            waveforms = [torchaudio_waveform(upstream / path, 16000).numpy()
                         for path in rows.iloc[start:start + 8].local_relpath]
            batch = extractor(waveforms, sampling_rate=16000, return_tensors="pt")["input_values"].to(device)
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                parts.append(model(input_values=batch).pooler_output.float().cpu().numpy())
            print(f"SSAST {min(start + 8, len(rows))}/{len(rows)}", flush=True)
        return np.concatenate(parts)
    if name == "aves":
        from aves import load_feature_extractor
        model = load_feature_extractor(
            str(weights / "aves-base-bio.torchaudio.model_config.json"),
            str(weights / "aves-base-bio.torchaudio.pt"), device=str(device), for_inference=True,
        ).eval()
        rate, minimum = 16000, 400
    else:
        model = cnn14_class()(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
                              fmin=50, fmax=14000, classes_num=527)
        checkpoint = torch.load(weights / "Cnn14_mAP=0.431.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        model = model.to(device).eval()
        rate, minimum = 32000, 9920
    model.requires_grad_(False)
    parts = []
    for i, path in enumerate(rows.local_relpath):
        waveform = torchaudio_waveform(upstream / path, rate, minimum).unsqueeze(0).to(device)
        with torch.inference_mode():
            embedding = (model.extract_features(waveform, layers=-1).mean(dim=1) if name == "aves"
                         else model(waveform, None)["embedding"])
        parts.append(embedding.float().cpu().numpy())
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"{name} {i + 1}/{len(rows)}", flush=True)
    return np.concatenate(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, help="Existing flat AudioCropped directory, if stored separately")
    parser.add_argument("--kind", required=True, choices=("ast", "acoustic", "cnn14", "aves", "ssast"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/cache"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    torch.set_num_threads(10)
    device = torch.device(args.device)
    if args.kind != "acoustic" and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    verify_source(args.root, args.upstream, author_features=False, audio_dir=args.audio_dir)
    weights_spec = None
    if args.kind != "acoustic":
        if args.weights is None:
            raise ValueError("Provide the pinned pretrained --weights directory")
        weights_spec = verify_weights(args.root, args.kind, args.weights)
    rows = load_clean_manifest(args.root / "data/manifest.csv")
    if args.audio_dir is not None:
        rows["local_relpath"] = [str((args.audio_dir / name).resolve()) for name in rows.filename]
    filenames = (["ast_fbank.npz", "ast_embeddings.npz", "block10_tokens.npy", "block10_index.npz"]
                 if args.kind == "ast" else [f"{args.kind}.npz"])
    for name in [*filenames, f"{args.kind}_extraction.json"]:
        if (args.output / name).exists():
            raise FileExistsError(f"Use a fresh output directory; existing cache retained: {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.kind == "ast":
        extract_ast(rows, args.upstream, args.weights, args.output, device)
    elif args.kind == "acoustic":
        values = []
        for i, path in enumerate(rows.local_relpath):
            values.append(extract_call_features(load_audio(args.upstream / path)))
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                print(f"Acoustic features {i + 1}/{len(rows)}", flush=True)
        np.savez_compressed(args.output / "acoustic.npz", features=np.stack(values),
                            feature_names=np.asarray(FEATURE_NAMES), **identities(rows))
    else:
        embeddings = extract_baseline(args.kind, rows, args.upstream, args.weights, device)
        dimensions = 2048 if args.kind == "cnn14" else 768
        if embeddings.shape != (len(rows), dimensions) or not np.isfinite(embeddings).all():
            raise ValueError("Invalid extracted embeddings")
        np.savez_compressed(args.output / f"{args.kind}.npz",
                            embeddings=embeddings.astype(np.float32), **identities(rows))
    write_json(args.output / f"{args.kind}_extraction.json", {
        "status": "complete", "kind": args.kind, "calls": len(rows), "cpu_threads": 10,
        "device": str(device), "torch": str(torch.__version__), "weights": weights_spec,
        "manifest_sha256": sha256(args.root / "data/manifest.csv"),
        "files": {name: sha256(args.output / name) for name in filenames},
        "note": "Array values, ordering and precision are reproduction targets; NPZ container bytes may differ.",
    })


if __name__ == "__main__":
    main()
