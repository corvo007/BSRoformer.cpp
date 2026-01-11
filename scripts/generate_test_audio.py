"""
Generate synthetic test audio for CI testing.

Creates deterministic test signals without requiring any external audio files,
ensuring reproducibility and avoiding copyright concerns.

Usage:
    python generate_test_audio.py --output test_audio.wav
"""

import argparse
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None


def generate_test_audio(
    output_path: str,
    duration: float = 5.0,
    sample_rate: int = 44100,
) -> None:
    """
    Generate deterministic test audio (sine wave synthesis).

    Creates a mixture of "vocal-like" and "accompaniment-like" sine waves
    that covers a reasonable frequency range for testing audio separation.

    Args:
        output_path: Path to save the output WAV file
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    """
    if sf is None:
        raise ImportError(
            "soundfile is required for audio generation. "
            "Install with: pip install soundfile"
        )

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # Simulate vocals: multiple sine waves (fundamental + harmonics)
    # Using A3 (220 Hz) as base frequency
    vocals = (
        0.50 * np.sin(2 * np.pi * 220 * t)  # A3 fundamental
        + 0.30 * np.sin(2 * np.pi * 440 * t)  # A4 harmonic
        + 0.15 * np.sin(2 * np.pi * 880 * t)  # A5 harmonic
        + 0.05 * np.sin(2 * np.pi * 1760 * t)  # A6 harmonic
    )

    # Add slight vibrato to vocals (more realistic)
    vibrato = 0.02 * np.sin(2 * np.pi * 5 * t)  # 5 Hz vibrato
    vocals = vocals * (1 + vibrato)

    # Simulate accompaniment: different frequency content
    accompaniment = (
        0.40 * np.sin(2 * np.pi * 110 * t)  # A2 bass
        + 0.30 * np.sin(2 * np.pi * 330 * t)  # E4
        + 0.20 * np.sin(2 * np.pi * 660 * t)  # E5
        + 0.10 * np.sin(2 * np.pi * 82.41 * t)  # E2 sub-bass
    )

    # Add slight amplitude envelope to make it more interesting
    envelope = np.ones_like(t)
    fade_samples = int(0.1 * sample_rate)  # 100ms fade
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

    # Mix vocals and accompaniment
    mix = (vocals + accompaniment) * envelope

    # Normalize to prevent clipping
    max_val = np.max(np.abs(mix))
    if max_val > 0:
        mix = mix / max_val * 0.9  # Leave some headroom

    # Create stereo (identical channels for simplicity)
    stereo = np.stack([mix, mix], axis=-1)

    # Save as WAV
    sf.write(output_path, stereo, sample_rate, subtype="PCM_16")
    print(f"Generated: {output_path}")
    print(f"  Duration: {duration}s")
    print(f"  Sample rate: {sample_rate} Hz")
    print("  Channels: 2 (stereo)")
    print("  Format: PCM_16")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic test audio for CI testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script generates deterministic test audio using sine wave synthesis.
The output is suitable for testing audio processing pipelines without
requiring real audio files.

Example:
    python generate_test_audio.py --output test.wav
    python generate_test_audio.py --output test.wav --duration 10 --sample-rate 48000
""",
    )

    parser.add_argument("--output", "-o", required=True, help="Output WAV file path")
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=5.0,
        help="Duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--sample-rate",
        "-sr",
        type=int,
        default=44100,
        help="Sample rate in Hz (default: 44100)",
    )

    args = parser.parse_args()

    generate_test_audio(
        output_path=args.output,
        duration=args.duration,
        sample_rate=args.sample_rate,
    )


if __name__ == "__main__":
    main()
