"""
visualise.py — Spectrogram generation & comparison plots.

Provides helpers to produce publication‑ready spectrogram images that
visually demonstrate the effect of the denoising pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def plot_spectrogram(
    file_path: str,
    *,
    n_fft: int = 2048,
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot the spectrogram of a WAV file.

    Parameters
    ----------
    file_path : str
        Path to a WAV audio file.
    n_fft : int
        FFT window size.
    title : str | None
        Plot title.  Defaults to ``"Spectrogram — <filename>"``.
    save_path : str | None
        If given, save the figure to this path (PNG recommended).
    show : bool
        Whether to call ``plt.show()``.
    """
    y, sr = librosa.load(file_path, sr=None)
    hop_length = n_fft // 4
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    s_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(
        s_db, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="hz", cmap="magma", ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title or f"Spectrogram — {Path(file_path).name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"✓  Spectrogram saved → {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def compare_spectrograms(
    original_path: str,
    cleaned_path: str,
    *,
    n_fft: int = 2048,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot before/after spectrograms side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    for ax, fpath, label in zip(
        axes,
        [original_path, cleaned_path],
        ["Before (Noisy)", "After (AI Denoised)"],
    ):
        y, sr = librosa.load(fpath, sr=None)
        hop = n_fft // 4
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        s_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        img = librosa.display.specshow(
            s_db, sr=sr, hop_length=hop,
            x_axis="time", y_axis="hz", cmap="magma", ax=ax,
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title(label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    fig.suptitle("AI Audio Denoiser — Spectrogram Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"✓  Comparison saved → {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
