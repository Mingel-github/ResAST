"""Twenty call-level F0, periodicity, energy and spectral descriptors."""

from dataclasses import dataclass
import math

import librosa
import numpy as np

FEATURE_NAMES = (
    "log_f0_q10", "log_f0_median", "log_f0_q90", "log_f0_iqr",
    "log_f0_std", "log_f0_mad", "log_f0_slope", "log_f0_abs_delta_median",
    "period_variation_proxy", "voiced_fraction", "voiced_probability_mean",
    "voiced_probability_std", "periodicity_median", "hnr_db_median",
    "amplitude_variation_proxy", "log_rms_median", "log_rms_iqr",
    "spectral_tilt_median", "spectral_tilt_iqr", "spectral_flatness_median",
)


@dataclass(frozen=True)
class AcousticSettings:
    frame_length: int = 1024
    hop_length: int = 160
    fmin_hz: float = 60.0
    fmax_hz: float = 2000.0


def finite_quantile(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, q)) if len(finite) else float("nan")


def finite_median(values: np.ndarray) -> float:
    return finite_quantile(values, 0.5)


def consecutive_pair_mask(valid: np.ndarray) -> np.ndarray:
    valid = np.asarray(valid, dtype=bool)
    if len(valid) < 2:
        return np.zeros(0, dtype=bool)
    return valid[:-1] & valid[1:]


def frame_periodicity(
    waveform: np.ndarray, f0: np.ndarray, sample_rate: int,
    frame_length: int, hop_length: int,
) -> np.ndarray:
    padded = np.pad(waveform, (frame_length // 2, frame_length // 2))
    result = np.full(len(f0), np.nan, dtype=np.float64)
    window = np.hanning(frame_length).astype(np.float64)
    for frame_index, frequency in enumerate(f0):
        if not np.isfinite(frequency) or frequency <= 0:
            continue
        lag = int(round(sample_rate / float(frequency)))
        if lag < 1 or lag >= frame_length // 2:
            continue
        start = frame_index * hop_length
        frame = padded[start : start + frame_length].astype(np.float64, copy=False)
        if len(frame) != frame_length:
            continue
        frame = (frame - frame.mean()) * window
        left, right = frame[:-lag], frame[lag:]
        denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
        if denominator <= 1.0e-12:
            continue
        result[frame_index] = float(np.clip(np.dot(left, right) / denominator, 0, 0.999))
    return result


def extract_call_features(
    waveform: np.ndarray, sample_rate: int = 16_000,
    settings: AcousticSettings = AcousticSettings(),
) -> np.ndarray:
    if waveform.ndim != 1 or not waveform.size:
        raise ValueError("Expected a nonempty mono waveform")
    frame_length, hop_length = settings.frame_length, settings.hop_length
    f0, _, voiced_probability = librosa.pyin(
        waveform, fmin=settings.fmin_hz, fmax=settings.fmax_hz,
        sr=sample_rate, frame_length=frame_length, hop_length=hop_length, center=True,
    )
    f0 = np.asarray(f0, dtype=np.float64)
    voiced_probability = np.asarray(voiced_probability, dtype=np.float64)
    magnitude = np.abs(librosa.stft(
        waveform, n_fft=frame_length, hop_length=hop_length,
        win_length=frame_length, center=True,
    )).astype(np.float64)
    rms = librosa.feature.rms(S=magnitude, frame_length=frame_length)[0]
    flatness = librosa.feature.spectral_flatness(S=magnitude + 1.0e-10)[0]
    frame_count = min(len(f0), magnitude.shape[1], len(rms), len(flatness))
    f0, voiced_probability = f0[:frame_count], voiced_probability[:frame_count]
    magnitude = magnitude[:, :frame_count]
    rms = np.asarray(rms[:frame_count], dtype=np.float64)
    flatness = np.asarray(flatness[:frame_count], dtype=np.float64)
    voiced = np.isfinite(f0) & (f0 > 0)
    log_f0 = np.full(frame_count, np.nan, dtype=np.float64)
    log_f0[voiced] = np.log(f0[voiced])
    q10, q25, q50, q75, q90 = (
        finite_quantile(log_f0, q) for q in (0.10, 0.25, 0.50, 0.75, 0.90)
    )
    finite_log_f0 = log_f0[voiced]
    log_std = float(np.std(finite_log_f0)) if len(finite_log_f0) else float("nan")
    log_mad = (
        float(np.median(np.abs(finite_log_f0 - np.median(finite_log_f0))))
        if len(finite_log_f0) else float("nan")
    )
    if voiced.sum() >= 2:
        time_axis = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)[voiced]
        log_slope = float(np.polyfit(time_axis, finite_log_f0, 1)[0])
    else:
        log_slope = float("nan")
    pair_mask = consecutive_pair_mask(voiced)
    log_delta = np.abs(np.diff(log_f0))[pair_mask]
    periods = np.full(frame_count, np.nan, dtype=np.float64)
    periods[voiced] = 1.0 / f0[voiced]
    period_delta = np.abs(np.diff(periods))[pair_mask]
    period_median = finite_median(periods)
    period_variation = (
        finite_median(period_delta) / period_median
        if np.isfinite(period_median) and period_median > 0 else float("nan")
    )
    periodicity = frame_periodicity(
        waveform, f0, sample_rate, frame_length, hop_length
    )[:frame_count]
    valid_periodicity = periodicity[np.isfinite(periodicity)]
    hnr = np.full_like(valid_periodicity, np.nan)
    if len(valid_periodicity):
        clipped = np.clip(valid_periodicity, 1.0e-6, 0.999)
        hnr = 10.0 * np.log10(clipped / (1.0 - clipped))
    log_rms = np.full(frame_count, np.nan, dtype=np.float64)
    log_rms[voiced] = np.log(rms[voiced] + 1.0e-10)
    rms_pair_delta = np.abs(np.diff(rms))[pair_mask]
    rms_median = finite_median(rms[voiced])
    amplitude_variation = (
        finite_median(rms_pair_delta) / rms_median
        if np.isfinite(rms_median) and rms_median > 0 else float("nan")
    )
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=frame_length)
    frequency_mask = (frequencies >= 200.0) & (frequencies <= 4000.0)
    log_frequency = np.log(frequencies[frequency_mask])
    centered_frequency = log_frequency - log_frequency.mean()
    denominator = float(np.dot(centered_frequency, centered_frequency))
    log_magnitude = np.log(magnitude[frequency_mask] + 1.0e-10)
    centered_magnitude = log_magnitude - log_magnitude.mean(axis=0, keepdims=True)
    spectral_tilt = (centered_frequency[:, None] * centered_magnitude).sum(axis=0) / denominator
    voiced_tilt, voiced_flatness = spectral_tilt[voiced], flatness[voiced]
    finite_voicing = voiced_probability[np.isfinite(voiced_probability)]
    return np.asarray([
        q10, q50, q90,
        q75 - q25 if np.isfinite(q75) and np.isfinite(q25) else np.nan,
        log_std, log_mad, log_slope, finite_median(log_delta), period_variation,
        float(voiced.mean()) if frame_count else np.nan,
        float(np.mean(finite_voicing)) if len(finite_voicing) else np.nan,
        float(np.std(finite_voicing)) if len(finite_voicing) else np.nan,
        finite_median(valid_periodicity), finite_median(hnr), amplitude_variation,
        finite_median(log_rms),
        finite_quantile(log_rms, 0.75) - finite_quantile(log_rms, 0.25)
        if np.isfinite(finite_quantile(log_rms, 0.75))
        and np.isfinite(finite_quantile(log_rms, 0.25)) else np.nan,
        finite_median(voiced_tilt),
        finite_quantile(voiced_tilt, 0.75) - finite_quantile(voiced_tilt, 0.25)
        if np.isfinite(finite_quantile(voiced_tilt, 0.75))
        and np.isfinite(finite_quantile(voiced_tilt, 0.25)) else np.nan,
        finite_median(voiced_flatness),
    ], dtype=np.float32)
