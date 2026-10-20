"""Run a saved AST variant on one audio call."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch

from resast.data.acoustic import extract_call_features
from resast.data.audio import load_audio, make_feature_extractor, waveform_to_features
from resast.evaluation.metrics import CLASSES
from resast.models.ast import FrozenASTPrefix, load_backbone
from resast.models.classifier import MODEL_FLAGS
from resast.training.checkpoint import load_checkpoint
from resast.training.common import configure_determinism, sha256, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True, help="Pinned downloaded AST snapshot directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    configure_determinism()
    device = torch.device(args.device)
    backbone = load_backbone(args.backbone)
    model, metadata = load_checkpoint(args.checkpoint, device, backbone=backbone)
    if metadata["model"] not in MODEL_FLAGS:
        raise ValueError("Audio demo supports the four AST variants")
    # Match the filterbank CPU execution setting used by the reference cache.
    torch.set_num_threads(10)
    waveform = load_audio(args.audio)
    features = waveform_to_features(waveform, make_feature_extractor(args.backbone))
    acoustic = None
    if MODEL_FLAGS[metadata["model"]][1]:
        acoustic = torch.from_numpy(np.asarray(extract_call_features(waveform), dtype=np.float32)[None]).to(device)
    prefix = FrozenASTPrefix(backbone).to(device).eval()
    with torch.inference_mode():
        tokens = prefix(torch.from_numpy(features).to(device))
        mapping = torch.zeros(len(tokens), dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(tokens, mapping, 1, acoustic)
        probabilities = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    result = {
        "audio_sha256": sha256(args.audio), "checkpoint_sha256": sha256(args.checkpoint),
        "model": metadata["model"], "prediction_unit": "one_call",
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASSES)},
        "predicted_age_group": CLASSES[int(probabilities.argmax())],
    }
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
