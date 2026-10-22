"""Waveform preparation and normalized log-mel inputs."""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 20_480
HOP_SAMPLES = 10_240
CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
REVISION = "f826b80d28226b62986cc218e5cec390b1096902"


def load_audio(path: str | Path, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if sample_rate != target_rate:
        divisor = math.gcd(sample_rate, target_rate)
        waveform = resample_poly(
            waveform, target_rate // divisor, sample_rate // divisor
        ).astype(np.float32)
    if waveform.size == 0:
        raise ValueError(f"Empty audio: {path}")
    return waveform.astype(np.float32, copy=False)


def segment_waveform(
    waveform: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = HOP_SAMPLES,
) -> list[np.ndarray]:
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("Expected a nonempty mono waveform")
    if window_samples < 1 or hop_samples < 1:
        raise ValueError("Window and hop must be positive")
    if len(waveform) <= window_samples:
        return [waveform]
    starts = list(range(0, len(waveform) - window_samples + 1, hop_samples))
    # Anchor the last full window at the call end when the hop misses it.
    final_start = len(waveform) - window_samples
    if starts[-1] != final_start:
        starts.append(final_start)
    return [waveform[start : start + window_samples] for start in starts]


def make_feature_extractor(
    source: str | Path = CHECKPOINT,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = True,
):
    from transformers import ASTFeatureExtractor

    extractor = ASTFeatureExtractor.from_pretrained(
        str(source),
        revision=REVISION,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    extractor.max_length = 128
    extractor.num_mel_bins = 128
    return extractor


def waveform_to_features(waveform: np.ndarray, extractor) -> np.ndarray:
    """Return [segments, 128 time frames, 128 mel bins] in call order."""
    segments = segment_waveform(waveform)
    return extractor(
        segments, sampling_rate=SAMPLE_RATE, return_tensors="np"
    )["input_values"].astype(np.float32)


def prepare_features(manifest: pd.DataFrame, data_root: str | Path, extractor) -> dict:
    """Preserve manifest order and the segment-to-call index in a feature bundle."""
    features, mappings, counts = [], [], []
    labels = {"kitten": 0, "adult": 1, "senior": 2}
    data_root = Path(data_root)
    for index, row in enumerate(manifest.itertuples(index=False)):
        batch = waveform_to_features(load_audio(data_root / row.local_relpath), extractor)
        features.extend(batch)
        mappings.extend([index] * len(batch))
        counts.append(len(batch))
    if not features:
        raise ValueError("The manifest contains no calls")
    return {
        "features": np.stack(features),
        "segment_call_indices": np.asarray(mappings, dtype=np.int32),
        "segment_counts": np.asarray(counts, dtype=np.int16),
        "call_ids": manifest["filename"].to_numpy(dtype=str),
        "cat_ids": manifest["analysis_cat_id"].to_numpy(dtype=str),
        "labels": np.asarray([labels[v] for v in manifest["age_group_filename"]], dtype=np.int8),
        "durations": manifest["duration_seconds"].to_numpy(dtype=np.float32),
        "source_paths": manifest["source_path"].to_numpy(dtype=str),
    }


def average_frozen_segments(
    embeddings: np.ndarray, segment_call_indices: np.ndarray, segment_counts: np.ndarray
) -> np.ndarray:
    """Aggregate original-AST embeddings in float64, then store as float32."""
    if embeddings.ndim != 2 or segment_call_indices.shape != (len(embeddings),):
        raise ValueError("Embedding rows and segment mapping differ")
    call_count = len(segment_counts)
    if (
        call_count == 0 or np.any(segment_call_indices < 0)
        or np.any(segment_call_indices >= call_count) or np.any(segment_counts <= 0)
        or not np.array_equal(
            np.bincount(segment_call_indices, minlength=call_count), segment_counts
        )
    ):
        raise ValueError("Invalid segment counts")
    calls = np.zeros((call_count, embeddings.shape[1]), dtype=np.float64)
    np.add.at(calls, segment_call_indices, embeddings)
    calls /= segment_counts[:, None]
    return calls.astype(np.float32)
