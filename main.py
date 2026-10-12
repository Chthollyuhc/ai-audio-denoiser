#!/usr/bin/env python3
"""
main.py — CLI entry‑point for the AI Audio Denoiser.

Usage
-----
    # Denoise a file (defaults)
    python main.py --input data/audio.wav --output data/audio_clean.wav

    # Custom filter range & noise strength
    python main.py -i noisy.wav -o clean.wav --low 100 --high 6000 --strength 0.9

    # Also generate spectrogram comparison image
    python main.py -i data/audio.wav -o data/audio_clean.wav --compare assets/comparison.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.filter import denoise
from src.visualise import compare_spectrograms, plot_spectrogram


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ai-audio-denoiser",
        description="AI Audio Denoiser — Remove background noise from speech recordings "
                    "using Spectral Gating and adaptive signal processing.",
    )
    p.add_argument("-i", "--input", required=True, help="Path to the noisy WAV file.")
    p.add_argument("-o", "--output", required=True, help="Path for the cleaned WAV output.")
    p.add_argument("--low", type=float, default=80.0, help="Bandpass low cut‑off (Hz). Default: 80")
    p.add_argument("--high", type=float, default=8000.0, help="Bandpass high cut‑off (Hz). Default: 8000")
    p.add_argument("--strength", type=float, default=0.85,
                   help="Noise suppression strength (0.0–1.0). Default: 0.85")
    p.add_argument("--stationary", action="store_true",
                   help="Treat noise as stationary (constant). Default: non‑stationary.")
    p.add_argument("--compare", type=str, default=None,
                   help="If set, save a before/after spectrogram comparison image to this path.")
    p.add_argument("--spectrogram", type=str, default=None,
                   help="Save the cleaned‑audio spectrogram to this path.")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress progress messages.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found — {input_path}", file=sys.stderr)
        sys.exit(1)

    # --- Run denoising pipeline ---
    denoise(
        str(input_path),
        args.output,
        low_hz=args.low,
        high_hz=args.high,
        stationary=args.stationary,
        prop_decrease=args.strength,
        verbose=not args.quiet,
    )

    # --- Optional visualisations ---
    if args.compare:
        compare_spectrograms(
            str(input_path), args.output,
            save_path=args.compare, show=False,
        )

    if args.spectrogram:
        plot_spectrogram(args.output, save_path=args.spectrogram, show=False)

    if not args.quiet:
        print("\n🎉 Done!  Your cleaned audio is ready.")


if __name__ == "__main__":
    main()
