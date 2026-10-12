"""
filter.py — Core audio‑denoising engine.

Pipeline
--------
1. **Bandpass Filter** — A 6th‑order Butterworth filter retains the speech
   band (default 80 Hz – 8 000 Hz) while discarding low‑frequency rumble
   and high‑frequency hiss.

2. **Spectral Gating** — A data‑driven, non‑stationary noise estimator
   (powered by the *noisereduce* library) builds a per‑frame noise profile
   from the signal itself and subtracts it in the STFT domain.  The
   ``prop_decrease`` parameter controls how aggressively noise is removed;
   the default of 0.85 preserves 15 % of the estimated noise so that the
   output still sounds natural rather than "robotic".

3. **RMS Matching** — The root‑mean‑square energy of the cleaned signal is
   rescaled to match the original, preventing unintended loudness changes.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
from scipy.signal import butter, sosfiltfilt


# ---------------------------------------------------------------------------
#  Bandpass filter helpers
# ---------------------------------------------------------------------------

def _butter_bandpass(low_hz: float, high_hz: float, sr: int, order: int = 6):
    """Design a Butterworth bandpass filter (second‑order sections)."""
    nyq = 0.5 * sr
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")
    return sos


def apply_bandpass(
    data: np.ndarray,
    sr: int,
    low_hz: float = 80.0,
    high_hz: float = 8000.0,
    order: int = 6,
) -> np.ndarray:
    """Apply a zero‑phase Butterworth bandpass filter.

    Works for both mono (1‑D) and multi‑channel (2‑D, channels‑first) audio.
    """
    sos = _butter_bandpass(low_hz, high_hz, sr, order)
    if data.ndim == 1:
        return sosfiltfilt(sos, data).astype(data.dtype)
    return np.array([sosfiltfilt(sos, ch) for ch in data], dtype=data.dtype)


# ---------------------------------------------------------------------------
#  Spectral Gating
# ---------------------------------------------------------------------------

def apply_spectral_gating(
    data: np.ndarray,
    sr: int,
    stationary: bool = False,
    prop_decrease: float = 0.85,
) -> np.ndarray:
    """Estimate noise and suppress it via spectral gating.

    Parameters
    ----------
    data : np.ndarray
        Mono audio signal.
    sr : int
        Sample rate.
    stationary : bool
        If *True*, assume the noise profile is constant across the signal.
        Set to *False* (default) for real‑world recordings with varying
        background noise such as street traffic.
    prop_decrease : float
        Proportion by which the estimated noise is reduced.  ``1.0`` removes
        noise completely (may sound robotic); ``0.85`` retains 15 % for a
        more natural result.
    """
    return nr.reduce_noise(
        y=data,
        sr=sr,
        stationary=stationary,
        prop_decrease=prop_decrease,
    )


# ---------------------------------------------------------------------------
#  RMS matching
# ---------------------------------------------------------------------------

def match_rms(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    """Scale *processed* so its RMS energy matches *original*."""
    rms_orig = np.sqrt(np.mean(original ** 2))
    rms_proc = np.sqrt(np.mean(processed ** 2))
    if rms_proc > 0:
        return processed * (rms_orig / rms_proc)
    return processed


# ---------------------------------------------------------------------------
#  Full pipeline
# ---------------------------------------------------------------------------

def denoise(
    input_path: str,
    output_path: str,
    *,
    low_hz: float = 80.0,
    high_hz: float = 8000.0,
    stationary: bool = False,
    prop_decrease: float = 0.85,
    verbose: bool = True,
) -> str:
    """Run the complete AI Audio Denoiser pipeline.

    Parameters
    ----------
    input_path : str
        Path to the noisy WAV file.
    output_path : str
        Where to write the cleaned WAV file.
    low_hz, high_hz : float
        Bandpass filter cut‑off frequencies (Hz).
    stationary : bool
        Whether the noise profile is assumed to be stationary.
    prop_decrease : float
        Noise suppression strength (0.0 – 1.0).
    verbose : bool
        Print progress messages.

    Returns
    -------
    str
        The *output_path* for convenience.
    """
    if verbose:
        print(f"[1/4] Loading audio: {input_path}")
    data, sr = librosa.load(input_path, sr=None, mono=True)

    if verbose:
        print(f"[2/4] Applying Bandpass Filter ({low_hz} Hz – {high_hz} Hz)…")
    filtered = apply_bandpass(data, sr, low_hz=low_hz, high_hz=high_hz)

    if verbose:
        print("[3/4] Running Spectral Gating (noise estimation & suppression)…")
    denoised = apply_spectral_gating(filtered, sr, stationary=stationary, prop_decrease=prop_decrease)

    if verbose:
        print("[4/4] RMS Matching (loudness normalisation)…")
    final = match_rms(data, denoised)

    sf.write(output_path, final, sr)
    if verbose:
        print(f"✓  Saved cleaned audio → {output_path}")
    return output_path
