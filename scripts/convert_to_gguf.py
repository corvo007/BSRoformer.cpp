#!/usr/bin/env python3
"""
Convert Mel-Band-Roformer PyTorch checkpoint to GGUF format.

Supports quantization: FP32, FP16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1
Mixed Quantization: Keeps Norms/Biases as FP32 to avoid CUDA alignment issues.
"""

import os
import argparse
import torch
import numpy as np
import yaml
import librosa
from einops import repeat, reduce, rearrange
import gguf
from gguf.quants import quantize, GGMLQuantizationType


def generate_buffers(hparams):
    """
    Generate the buffers (freq_indices, num_bands_per_freq, etc.)
    mimicking the logic in MelBandRoformer.__init__.
    """
    num_bands = hparams["num_bands"]
    sample_rate = hparams.get("sample_rate", 44100)
    stft_n_fft = hparams.get("stft_n_fft", 2048)
    stereo = hparams.get("stereo", False)

    # 1. Calculate number of frequencies
    freqs = stft_n_fft // 2 + 1

    # 2. Create Mel Filter Bank
    mel_filter_bank_numpy = librosa.filters.mel(
        sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands
    )
    mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

    # 3. Ensure edge values are positive (required for mask generation)
    # The exact value doesn't matter as long as it's > 0
    mel_filter_bank[0, 0] = max(mel_filter_bank[0, 0].item(), 1e-6)
    mel_filter_bank[-1, -1] = max(mel_filter_bank[-1, -1].item(), 1e-6)

    # 4. Create Masks
    freqs_per_band = mel_filter_bank > 0
    assert freqs_per_band.any(dim=0).all(), (
        "all frequencies need to be covered by all bands"
    )

    # 5. Generate Indices
    repeated_freq_indices = repeat(torch.arange(freqs), "f -> b f", b=num_bands)
    freq_indices = repeated_freq_indices[freqs_per_band]

    if stereo:
        freq_indices = repeat(freq_indices, "f -> f s", s=2)
        # s=0 -> 2*f, s=1 -> 2*f+1
        freq_indices = freq_indices * 2 + torch.arange(2)
        freq_indices = rearrange(freq_indices, "f s -> (f s)")

    # 6. Aggregate Counts
    num_freqs_per_band = reduce(freqs_per_band, "b f -> b", "sum")
    num_bands_per_freq = reduce(freqs_per_band, "b f -> f", "sum")

    return {
        "freq_indices": freq_indices,
        "num_freqs_per_band": num_freqs_per_band,
        "num_bands_per_freq": num_bands_per_freq,
        "freqs_per_band": freqs_per_band,  # Kept if needed, though usually not saved
    }


# ============================================================================
# Quantization Helper
# ============================================================================


def get_target_quantization_type(dtype_str: str) -> GGMLQuantizationType:
    mapping = {
        "f32": GGMLQuantizationType.F32,
        "fp32": GGMLQuantizationType.F32,
        "f16": GGMLQuantizationType.F16,
        "fp16": GGMLQuantizationType.F16,
        "q8_0": GGMLQuantizationType.Q8_0,
        "q4_0": GGMLQuantizationType.Q4_0,
        "q4_1": GGMLQuantizationType.Q4_1,
        "q5_0": GGMLQuantizationType.Q5_0,
        "q5_1": GGMLQuantizationType.Q5_1,
    }
    return mapping.get(dtype_str.lower(), GGMLQuantizationType.F32)


def get_file_type_id(qtype: GGMLQuantizationType) -> int:
    # See GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
    mapping = {
        GGMLQuantizationType.F32: 0,
        GGMLQuantizationType.F16: 1,
        GGMLQuantizationType.Q4_0: 2,
        GGMLQuantizationType.Q4_1: 3,
        # 4 is Q4_1_O (deprecated/legacy?)
        # 5 is Q4_0_O ?
        # 6 is Q4_1_O ?
        GGMLQuantizationType.Q8_0: 7,
        GGMLQuantizationType.Q5_0: 8,
        GGMLQuantizationType.Q5_1: 9,
        GGMLQuantizationType.Q2_K: 10,
        GGMLQuantizationType.Q3_K: 11,
        GGMLQuantizationType.Q4_K: 12,
        GGMLQuantizationType.Q5_K: 13,
        GGMLQuantizationType.Q6_K: 14,
        # IQ2_XXS etc might have IDs but let's stick to these for now
    }
    return mapping.get(qtype, 0)  # Default to ALL_F32 if unknown


def should_quantize(name: str) -> bool:
    """
    Determine if a tensor should be quantized.
    Keep norms and biases as FP32 to avoid CUDA alignment issues.
    """
    # Biases are always small and sensitive
    if "bias" in name:
        return False

    # Norm weights (gamma) must be F32 to avoid mixed-type mul issues in CUDA
    if "norm.weight" in name:
        return False

    # Quantize all other "weight" matrices (Linear, Conv, Embedding if any)
    if "weight" in name:
        return True

    return False


# ============================================================================
# Key Name Mapping
# ============================================================================


def map_key_name(key: str) -> str:
    """
    Map PyTorch state_dict keys to GGUF format (blk.{bid}.*).
    Standardizes suffixes: gamma -> weight, beta -> bias.
    """

    def standardize_suffix(param_name: str) -> str:
        if param_name == "gamma":
            return "weight"
        if param_name == "beta":
            return "bias"
        return param_name

    parts = key.split(".")
    suffix = standardize_suffix(parts[-1])

    # Transformer Layers
    if key.startswith("layers."):
        layer_idx = parts[1]
        tf_idx = parts[2]  # 0=Time, 1=Freq
        type_str = "time" if tf_idx == "0" else "freq"

        # Final Norm: layers.0.0.norm.gamma
        if len(parts) >= 5 and parts[3] == "norm":
            return f"blk.{layer_idx}.{type_str}_norm.{suffix}"

        # Sub-layers (Attention=0, FF=1)
        if len(parts) >= 6 and parts[3] == "layers":
            block_sub_idx = parts[5]

            if block_sub_idx == "0":  # Attention
                if len(parts) > 6:
                    sub_name = parts[6]
                    if sub_name == "norm":
                        return f"blk.{layer_idx}.{type_str}_attn_norm.{suffix}"
                    if sub_name == "to_qkv":
                        return f"blk.{layer_idx}.{type_str}_attn_qkv.{suffix}"
                    if sub_name == "to_out":
                        return f"blk.{layer_idx}.{type_str}_attn_out.{suffix}"
                    if sub_name == "to_gates":
                        return f"blk.{layer_idx}.{type_str}_attn_gate.{suffix}"

            elif block_sub_idx == "1":  # FeedForward
                if len(parts) >= 8 and parts[6] == "net":
                    net_idx = parts[7]
                    if net_idx == "0":
                        return f"blk.{layer_idx}.{type_str}_ff_norm.{suffix}"
                    if net_idx == "1":
                        return f"blk.{layer_idx}.{type_str}_ff_in.{suffix}"
                    if net_idx == "4":
                        return f"blk.{layer_idx}.{type_str}_ff_out.{suffix}"

    # BandSplit
    if key.startswith("band_split.to_features"):
        band_idx = parts[2]
        layer_idx = parts[3]  # 0=Norm, 1=Linear

        if layer_idx == "0":
            return f"band_split.{band_idx}.norm.{suffix}"
        if layer_idx == "1":
            return f"band_split.{band_idx}.linear.{suffix}"

    # Mask Estimator
    if key.startswith("mask_estimators"):
        est_idx = parts[1]
        freq_idx = parts[3]
        layer_idx = parts[5]  # 0, 2, 4
        return f"mask_est.{est_idx}.freq.{freq_idx}.mlp.{layer_idx}.{suffix}"

    return key.replace(".", "_")


# ============================================================================
# Main Conversion
# ============================================================================


def convert(
    ckpt_path: str,
    output_path: str,
    config_path: str,
    dtype: str = "fp32",
    name: str = None,
    description: str = None,
):
    """
    Convert PyTorch checkpoint to GGUF format.
    """
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    print(f"Loading config: {config_path}")
    with open(config_path) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    # Generate buffers
    print("Generating buffers (standalone)...")
    buffers = generate_buffers(config_dict["model"])
    freq_indices = buffers["freq_indices"]
    num_bands_per_freq = buffers["num_bands_per_freq"]
    num_freqs_per_band = buffers["num_freqs_per_band"]

    # Create GGUF writer
    gguf_writer = gguf.GGUFWriter(output_path, "mel_band_roformer")

    # =========================================================================
    # 1. Write Standard GGUF Metadata
    # =========================================================================
    print("Writing metadata...")

    # General metadata
    model_name = name if name else "Mel-Band-Roformer Separator"
    model_description = description if description else "Music source separation model"
    gguf_writer.add_name(model_name)
    gguf_writer.add_description(model_description)

    # Determine types
    target_qtype = get_target_quantization_type(dtype)
    file_type_id = get_file_type_id(target_qtype)

    gguf_writer.add_file_type(file_type_id)

    # Quantization version (required when quantized)
    if target_qtype != GGMLQuantizationType.F32:
        gguf_writer.add_quantization_version(2)

    # Calculate parameter count
    total_params = 0
    for key, tensor in state_dict.items():
        if "freq_indices" in key or "num_bands" in key:
            continue
        total_params += tensor.numel()

    print(f"Total parameters: {total_params}")
    gguf_writer.add_uint64("general.parameter_count", total_params)

    # =========================================================================
    # 2. Write Hyperparameters
    # =========================================================================
    print("Writing hyperparameters...")
    hparams = config_dict["model"]

    # Architecture specific parameters
    gguf_writer.add_uint32("mel_band_roformer.dim", hparams["dim"])
    gguf_writer.add_uint32("mel_band_roformer.depth", hparams["depth"])
    gguf_writer.add_uint32("mel_band_roformer.num_bands", hparams["num_bands"])

    # STFT parameters
    gguf_writer.add_uint32(
        "mel_band_roformer.stft_n_fft", hparams.get("stft_n_fft", 2048)
    )
    # Remove default for hop_length, must be present or fail/warn
    gguf_writer.add_uint32(
        "mel_band_roformer.stft_hop_length", hparams.get("stft_hop_length", 441)
    )
    gguf_writer.add_uint32(
        "mel_band_roformer.stft_win_length", hparams.get("stft_win_length", 2048)
    )
    gguf_writer.add_bool(
        "mel_band_roformer.stft_normalized", hparams.get("stft_normalized", False)
    )
    gguf_writer.add_bool(
        "mel_band_roformer.zero_dc", hparams.get("zero_dc", True)
    )  # Defaults to True in reference implementation

    # Architecture details
    gguf_writer.add_uint32("mel_band_roformer.num_stems", hparams.get("num_stems", 1))
    gguf_writer.add_bool("mel_band_roformer.stereo", hparams.get("stereo", False))
    gguf_writer.add_uint32(
        "mel_band_roformer.sample_rate", hparams.get("sample_rate", 44100)
    )

    gguf_writer.add_uint32(
        "mel_band_roformer.time_transformer_depth",
        hparams.get("time_transformer_depth", 0),
    )
    gguf_writer.add_uint32(
        "mel_band_roformer.freq_transformer_depth",
        hparams.get("freq_transformer_depth", 0),
    )
    gguf_writer.add_uint32(
        "mel_band_roformer.linear_transformer_depth",
        hparams.get("linear_transformer_depth", 0),
    )

    gguf_writer.add_uint32(
        "mel_band_roformer.mask_estimator_depth", hparams.get("mask_estimator_depth", 1)
    )
    gguf_writer.add_uint32("mel_band_roformer.dim_head", hparams.get("dim_head", 64))
    gguf_writer.add_uint32("mel_band_roformer.heads", hparams.get("heads", 8))
    gguf_writer.add_uint32(
        "mel_band_roformer.mlp_expansion_factor", hparams.get("mlp_expansion_factor", 4)
    )
    gguf_writer.add_bool(
        "mel_band_roformer.skip_connection", hparams.get("skip_connection", False)
    )

    # =========================================================================
    # 3. Write Inference Defaults (Optional, can be overridden at runtime)
    # =========================================================================
    print("Writing inference defaults...")

    inference_config = config_dict.get("inference", {})
    audio_config = config_dict.get("audio", {})

    # chunk_size: prefer inference.chunk_size, fallback to audio.chunk_size
    default_chunk_size = inference_config.get(
        "chunk_size", audio_config.get("chunk_size", 352800)
    )
    # num_overlap: from inference section
    default_num_overlap = inference_config.get("num_overlap", 0)

    gguf_writer.add_uint32("mel_band_roformer.default_chunk_size", default_chunk_size)
    gguf_writer.add_uint32("mel_band_roformer.default_num_overlap", default_num_overlap)

    # =========================================================================
    # 4. Write Buffers (Always FP32/I32)
    # =========================================================================
    print("Writing buffers...")

    # freq_indices (int32)
    gguf_writer.add_tensor("buffer_freq_indices", freq_indices.numpy().astype(np.int32))
    # num_bands_per_freq (int32)
    gguf_writer.add_tensor(
        "buffer_num_bands_per_freq", num_bands_per_freq.numpy().astype(np.int32)
    )
    # num_freqs_per_band (int32)
    gguf_writer.add_tensor(
        "buffer_num_freqs_per_band", num_freqs_per_band.numpy().astype(np.int32)
    )

    # =========================================================================
    # 5. Write Weights (Mixed Quantization)
    # =========================================================================
    print(f"Writing weights ({dtype} -> {target_qtype.name})...")
    print("Strategy: Quantize weights, Keep Norm/Bias as F32")

    n_tensors = 0
    n_quantized = 0

    for key, tensor in state_dict.items():
        new_key = map_key_name(key)

        # Skip buffers
        if (
            "freq_indices" in key
            or "num_bands_per_freq" in key
            or "num_freqs_per_band" in key
        ):
            continue

        data = tensor.numpy().astype(np.float32)

        # Decide whether to quantize
        is_quantized = False

        if target_qtype != GGMLQuantizationType.F32 and should_quantize(new_key):
            try:
                # Use gguf-py built-in quantization
                quantized_data = quantize(data, target_qtype)
                # Pass raw_dtype so GGUFWriter knows how to treat the byte array (for Q types)
                # or float array (for F16)
                gguf_writer.add_tensor(new_key, quantized_data, raw_dtype=target_qtype)
                is_quantized = True
                n_quantized += 1
            except Exception as e:
                print(
                    f"Warning: Failed to quantize {new_key} to {target_qtype.name}, falling back to F32. Error: {e}"
                )
                gguf_writer.add_tensor(new_key, data)
        else:
            # Keep as F32
            gguf_writer.add_tensor(new_key, data)

        status = target_qtype.name if is_quantized else "F32"
        print(f"  {new_key:<50} | {str(data.shape):<20} | {status}")
        n_tensors += 1

    # =========================================================================
    # 6. Write File
    # =========================================================================
    print(f"\nWriting GGUF to {output_path}")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    file_size = os.path.getsize(output_path)
    print(f"\nDone! Converted {n_tensors} tensors ({n_quantized} quantized)")
    print(f"Output file size: {file_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Mel-Band-Roformer checkpoint to GGUF format with Mixed Quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_to_gguf.py --ckpt model.ckpt --config config.yaml --out model_f16.gguf --dtype fp16
  python convert_to_gguf.py --ckpt model.ckpt --config config.yaml --out model_q8.gguf --dtype q8_0
""",
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to PyTorch checkpoint"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--out", type=str, required=True, help="Output GGUF file path")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=[
            "fp32",
            "f32",
            "fp16",
            "f16",
            "q8_0",
            "q4_0",
            "q4_1",
            "q5_0",
            "q5_1",
        ],
        help="Target quantization type. Norms/Biases will be kept as F32. (K-Quants not supported due to dim=384)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Model name (default: 'Mel-Band-Roformer Vocal Separator')",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Model description (default: 'Audio source separation model for vocal extraction')",
    )
    args = parser.parse_args()

    convert(args.ckpt, args.out, args.config, args.dtype, args.name, args.description)
