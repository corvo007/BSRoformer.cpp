"""
Generate minimal test data for MelBandRoformer.cpp verification.

This script generates ONLY the essential tensors needed for C++ tests:
- input_audio.npy  (for test_inference)
- output_audio.npy (for test_inference)
- band_split_in.npy (for test_component_bandsplit)
- after_band_split.npy (for test_component_bandsplit, test_component_layers)
- before_mask_est.npy (for test_component_layers, test_component_mask)
- mask_est0.npy (for test_component_mask)
- chunk_in.npy (for test_chunking_logic)
- chunk_out.npy (for test_chunking_logic)

Requirements:
    This script requires the Music-Source-Separation-Training repository:
    https://github.com/ZFTurbo/Music-Source-Separation-Training

    Clone it first:
        git clone https://github.com/ZFTurbo/Music-Source-Separation-Training.git

Usage:
    python generate_test_data.py --model-repo /path/to/Music-Source-Separation-Training \\
        --audio test.wav --checkpoint model.ckpt --output test_data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import yaml
from ml_collections import ConfigDict
from einops import rearrange, pack, unpack

# Model imports are deferred until we know the model-repo path
# Model imports are deferred until we know the model-repo path
MelBandRoformer = None
BSRoformer = None
pack_one = None
unpack_one = None
# Inference utility
inference_func = None

MODEL_REPO_URL = "https://github.com/ZFTurbo/Music-Source-Separation-Training"


class MockModel(torch.nn.Module):
    """Identity model for testing chunking logic."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: [Batch, Channels, Time] or [Batch, Time]
        # Return same as input (Identity)
        return x


def load_model_module(model_repo_path: Path):
    """Dynamically load the MelBandRoformer model from the specified repository."""
    global MelBandRoformer, BSRoformer, pack_one, unpack_one, inference_func

    if not model_repo_path.exists():
        print("\n" + "=" * 70)
        print("ERROR: Model repository not found!")
        print("=" * 70)
        print(f"\nPath: {model_repo_path}")
        print("\nThis script requires the Music-Source-Separation-Training repository.")
        print("\nPlease clone it first:")
        print(f"  git clone {MODEL_REPO_URL}")
        print(
            "\nThen run this script with --model-repo pointing to the cloned directory."
        )
        print("=" * 70)
        sys.exit(1)

    models_path = model_repo_path / "models"
    if not models_path.exists():
        print("\n" + "=" * 70)
        print("ERROR: Invalid repository structure!")
        print("=" * 70)
        print(f"\nThe 'models' directory was not found in: {model_repo_path}")
        print("=" * 70)
        sys.exit(1)

    # Add to path and import
    sys.path.insert(0, str(model_repo_path))

    # Mock loralib to allow importing model_utils without installing it
    from unittest.mock import MagicMock

    if "loralib" not in sys.modules:
        sys.modules["loralib"] = MagicMock()

    # Import from new structure (Music-Source-Separation-Training)
    try:
        from models.bs_roformer.mel_band_roformer import (
            MelBandRoformer as _MelBandRoformer,
        )
        from models.bs_roformer.mel_band_roformer import (
            pack_one as _pack_one,
            unpack_one as _unpack_one,
        )

        pack_one = _pack_one
        unpack_one = _unpack_one
        MelBandRoformer = _MelBandRoformer

        try:
            from models.bs_roformer.bs_roformer import BSRoformer as _BSRoformer

            BSRoformer = _BSRoformer
        except ImportError:
            print("  Warning: Could not import BSRoformer from model repo.")

        # Import demix from utils.model_utils
        from utils.model_utils import demix

        inference_func = demix

        print(f"  Loaded model from: {model_repo_path}")
        return
    except ImportError as e:
        print("\n" + "=" * 70)
        print("ERROR: Failed to import model!")
        print("=" * 70)
        print(f"\nImport error: {e}")
        print(
            "\nPlease ensure the repository is complete and dependencies are installed."
        )
        sys.exit(1)


def save_tensor(
    output_dir: Path, name: str, tensor, subdir: str = "activations"
) -> dict:
    """Save tensor to .npy file."""
    path = output_dir / subdir / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
        if tensor.dtype in [torch.int64, torch.int32, torch.bool]:
            tensor = tensor.float()
        tensor = tensor.numpy()

    if isinstance(tensor, np.ndarray) and tensor.dtype != np.float32:
        tensor = tensor.astype(np.float32)

    np.save(path, tensor)
    print(f"  Saved {name}: shape={list(tensor.shape)}")
    return {"name": name, "shape": list(tensor.shape), "path": str(path)}


def generate_chunking_data(output_dir: Path, config: ConfigDict):
    """Generate input/output data for verifying chunking logic."""
    print("\n[Chunking] Generating overlap-add debug data...")

    if inference_func is None:
        print(
            "  Warning: Inference function not found, skipping chunking data generation."
        )
        return

    # Create Mock Model (Identity)
    model = MockModel()
    device = torch.device("cpu")

    # Create input: Ramp signal
    # Size > 2 chunks to test overlap logic
    # Use fixed values to match C++ test_chunking_logic.cpp (lines 76-77)
    chunk_size = 352800
    num_overlap = 2

    print(f"  Chunk size: {chunk_size}, Overlap: {num_overlap}")

    total_len = chunk_size * 2 + 10000
    inputs = np.linspace(0, 1, total_len).astype(np.float32)
    # Make stereo [2, T]
    inputs = np.stack([inputs, inputs], axis=0)

    # Save input (C-order, transposed to [T, 2] for C++ ease if needed, but C++ load_npy handles it)
    save_tensor(output_dir, "chunk_in", inputs.T, subdir=".")

    # Run Inference
    mixture = torch.tensor(inputs, dtype=torch.float32)

    # demix(config, model, mix, device, model_type)
    # generic mode (not htdemucs) uses 'generic'
    # It returns dict {instr: waveform} or array
    res = inference_func(config, model, mixture, device, model_type="generic")

    if isinstance(res, dict):
        # Pick the first instrument
        first_key = list(res.keys())[0]
        output = res[first_key]
    else:
        output = res

    # Save output
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    save_tensor(output_dir, "chunk_out", output.T, subdir=".")


def generate_test_data(
    model_repo: str,
    audio_file: str,
    checkpoint: str,
    config_file: str,
    output_dir: str,
    audio_start: float = 2.0,
    audio_end: float = 5.0,
) -> int:
    """Generate test data for C++ verification."""

    # Load model module from specified repository
    model_repo_path = Path(model_repo)
    load_model_module(model_repo_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MelBandRoformer Test Data Generator")
    print("=" * 70)

    # 1. Load config and model
    print(f"\n[1/4] Loading model from {checkpoint}")

    with open(config_file) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    model_type = "mel_band"
    if "freqs_per_bands" in config.model:
        model_type = "bs"
        if BSRoformer is None:
            print(
                "Error: BSRoformer class not loaded but config looks like BS Roformer."
            )
            return 1
        model = BSRoformer(**dict(config.model))
        print(f"  Architecture: Band Split Roformer")
    else:
        model = MelBandRoformer(**dict(config.model))
        print(f"  Architecture: Mel-Band Roformer")

    state_dict = torch.load(checkpoint, map_location="cpu")
    # Handle checkpoint structure
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict)
    model.eval()

    print(f"  Config: depth={config.model.depth}, dim={config.model.dim}")

    # 2. Load audio
    print(f"\n[2/4] Loading audio ({audio_start}s - {audio_end}s) from {audio_file}")

    audio, sr = sf.read(audio_file)
    start_sample = int(audio_start * sr)
    end_sample = int(audio_end * sr)
    audio_segment = audio[start_sample:end_sample]

    if len(audio_segment.shape) == 1:
        audio_segment = np.stack([audio_segment, audio_segment], axis=-1)

    # [batch, channels, samples]
    audio_tensor = torch.tensor(audio_segment.T, dtype=torch.float32).unsqueeze(0)
    print(f"  Audio shape: {audio_tensor.shape}")

    # 3. Run instrumented forward pass
    print("\n[3/4] Running instrumented forward pass...")

    captured = {}

    with torch.no_grad():
        device = audio_tensor.device
        raw_audio = audio_tensor

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")

        batch, channels, raw_audio_length = raw_audio.shape
        istft_length = raw_audio_length

        # STFT
        raw_audio_packed, batch_audio_channel_packed_shape = pack_one(raw_audio, "* t")
        stft_window = model.stft_window_fn(device=device)
        stft_repr = torch.stft(
            raw_audio_packed,
            **model.stft_kwargs,
            window=stft_window,
            return_complex=True,
        )
        stft_repr = torch.view_as_real(stft_repr)

        # ===== CAPTURE: Raw STFT/ISTFT for C++ Verification =====
        # Unpack to [batch, channels, freq, time, 2]
        stft_raw_unpacked = unpack_one(
            stft_repr, batch_audio_channel_packed_shape, "* f t c"
        )
        captured["stft_raw"] = stft_raw_unpacked.clone()

        # Compute ISTFT directly on this raw STFT (Identity check)
        stft_complex = torch.view_as_complex(stft_repr)
        istft_check = torch.istft(
            stft_complex,
            **model.stft_kwargs,
            window=stft_window,
            return_complex=False,
            length=istft_length,
        )
        istft_check_unpacked = unpack_one(
            istft_check, batch_audio_channel_packed_shape, "* t"
        )
        captured["istft_raw"] = istft_check_unpacked.clone()
        # ========================================================

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, "* f t c")
        stft_repr = rearrange(stft_repr, "b s f t c -> b (f s) t c")

        # Frequency indexing
        if model_type == "mel_band":
            batch_arange = torch.arange(batch, device=device)[..., None]
            x = stft_repr[batch_arange, model.freq_indices]
            x = rearrange(x, "b f t c -> b t (f c)")
        else:
            # BS Roformer: Direct usage
            x = stft_repr
            # If stft_repr is complex (view_as_real result: [b, f, t, 2])
            # BS model expects: [b, f, t, 2] -> rearrange to [b, t, (f * 2)]
            # Wait, bs_roformer.py: x = rearrange(x, 'b f t c -> b t (f c)')
            x = rearrange(x, "b f t c -> b t (f c)")

        # ===== CAPTURE: BandSplit Input =====
        captured["band_split_in"] = x.clone()

        # BandSplit
        x = model.band_split(x)

        # ===== CAPTURE: After BandSplit (= Transformer Input) =====
        captured["after_band_split"] = x.clone()

        # Transformer Layers
        for layer_idx, (time_transformer, freq_transformer) in enumerate(model.layers):
            # Time Transformer
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = time_transformer(x)
            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")

            # Freq Transformer
            x, ps = pack([x], "* f d")
            x = freq_transformer(x)
            (x,) = unpack(x, ps, "* f d")

        # BS Roformer: Apply global final_norm after all transformer layers
        if model_type == "bs" and hasattr(model, "final_norm"):
            x = model.final_norm(x)

        # ===== CAPTURE: Before Mask Estimator (= Transformer Output) =====
        captured["before_mask_est"] = x.clone()

        # Mask Estimator (just first one for testing)
        mask0 = model.mask_estimators[0](x)

        # ===== CAPTURE: Mask Estimator Output =====
        captured["mask_est0"] = mask0.clone()

        # Continue with full forward pass for output
        num_stems = len(model.mask_estimators)
        masks = torch.stack([fn(x) for fn in model.mask_estimators], dim=1)
        masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)

        stft_repr = rearrange(stft_repr, "b f t c -> b 1 f t c")
        stft_repr = torch.view_as_complex(stft_repr)
        masks = torch.view_as_complex(masks)
        masks = masks.type(stft_repr.dtype)

        from einops import repeat

        if model_type == "mel_band":
            scatter_indices = repeat(
                model.freq_indices,
                "f -> b n f t",
                b=batch,
                n=num_stems,
                t=stft_repr.shape[-1],
            )
            stft_repr_expanded_stems = repeat(
                stft_repr, "b 1 ... -> b n ...", n=num_stems
            )
            masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(
                2, scatter_indices, masks
            )

            denom = repeat(model.num_bands_per_freq, "f -> (f r) 1", r=channels)
            masks_averaged = masks_summed / denom.clamp(min=1e-8)

            stft_repr = stft_repr * masks_averaged

        else:
            # BS Roformer: Direct mask application
            # masks shape: [b, n, f, t, c] (rearranged above)
            # stft_repr shape: [b, 1, f, t, c] (rearranged above)

            # BS model output masks are often [b, n, f, t] (complex/real?)
            # Wait, bs_roformer.py:
            # masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
            # masks = rearrange(masks, 'b n t (f c) -> b n f t c', c = 2)
            # x = x * masks.sum(dim=1) # summation over stems? No, output separate stems.
            # return x * masks

            # So here: stft_repr * masks is correct.
            stft_repr = stft_repr * masks

        # ISTFT
        if model_type == "mel_band":
            stft_repr = rearrange(
                stft_repr, "b n (f s) t -> (b n s) f t", s=model.audio_channels
            )
        else:
            # BS Roformer: stft_repr is [b, n, (Freq*Stereo), t] (complex)
            # Unpack stereo and flatten batch/stems/stereo for istft
            stft_repr = rearrange(
                stft_repr, "b n (f s) t -> (b n s) f t", s=model.audio_channels
            )

        if getattr(model, "zero_dc", False):
            # Zero out DC component
            stft_repr = stft_repr.clone()
            stft_repr[:, 0, :] = 0.0

        recon_audio = torch.istft(
            stft_repr,
            **model.stft_kwargs,
            window=stft_window,
            return_complex=False,
            length=istft_length,
        )
        recon_audio = rearrange(
            recon_audio,
            "(b n s) t -> b n s t",
            b=batch,
            s=model.audio_channels,
            n=num_stems,
        )

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, "b 1 s t -> b s t")
            captured["output_audio"] = recon_audio.clone()
        else:
            # Capture Stem 0 for verification
            captured["output_audio"] = recon_audio[:, 0, :, :].clone()

            # Capture Stem 0 for verification
            captured["output_audio"] = recon_audio[:, 0, :, :].clone()

    # 4. Generate Chunking Debug Data
    generate_chunking_data(output_path, config)

    # 5. Save tensors
    print(f"\n[4/5] Saving test data to {output_dir}")

    # Input audio
    save_tensor(output_path, "input_audio", audio_tensor)

    # Captured tensors
    for name, tensor in captured.items():
        save_tensor(output_path, name, tensor)

    # Verify outputs match normal forward pass
    print("\n[Verification] Checking output matches model.forward()...")
    with torch.no_grad():
        baseline = model(audio_tensor)
        if hasattr(model, "num_stems") and model.num_stems > 1:
            baseline = baseline[:, 0, :, :]

    diff = (baseline - captured["output_audio"]).abs()
    max_diff = diff.max().item()

    if max_diff > 1e-6:
        print(f"  ✗ FAILED: max_diff = {max_diff:.2e}")
        return 1
    else:
        print(f"  ✓ PASSED: max_diff = {max_diff:.2e}")

    print("\n" + "=" * 70)
    print("Test data generation complete!")
    print(f"  Output: {output_dir}/activations/")
    print(f"  Files: {len(captured) + 1} tensors")
    print("=" * 70)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate test data for BSRoformer.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Requirements:
  This script requires the original Mel-Band-Roformer-Vocal-Model repository.
  
  Clone it first:
    git clone {MODEL_REPO_URL}
  
  Then specify the path with --model-repo.

Example:
  python generate_test_data.py \\
    --model-repo /path/to/Mel-Band-Roformer-Vocal-Model \\
    --audio test.wav \\
    --checkpoint model.ckpt \\
    --output test_data
""",
    )

    parser.add_argument(
        "--model-repo",
        required=True,
        help=f"Path to Mel-Band-Roformer-Vocal-Model repository (clone from {MODEL_REPO_URL})",
    )
    parser.add_argument("--audio", required=True, help="Input audio file (WAV)")
    parser.add_argument(
        "--checkpoint", required=True, help="Model checkpoint file (.ckpt)"
    )
    parser.add_argument(
        "--config",
        help="Model config YAML file (default: <model-repo>/configs/config_vocals_mel_band_roformer.yaml)",
    )
    parser.add_argument(
        "--output", default="test_data", help="Output directory for test data"
    )
    parser.add_argument(
        "--start",
        type=float,
        default=2.0,
        help="Audio start time in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=5.0,
        help="Audio end time in seconds (default: 5.0)",
    )

    args = parser.parse_args()

    # Resolve paths
    model_repo_path = Path(args.model_repo).resolve()
    audio_path = Path(args.audio).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()

    # Config defaults to model-repo/configs/...
    if args.config:
        config_path = Path(args.config).resolve()
    else:
        config_path = (
            model_repo_path / "configs" / "config_vocals_mel_band_roformer.yaml"
        )

    # Validate paths
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        return 1

    return generate_test_data(
        str(model_repo_path),
        str(audio_path),
        str(checkpoint_path),
        str(config_path),
        str(output_path),
        args.start,
        args.end,
    )


if __name__ == "__main__":
    sys.exit(main())
