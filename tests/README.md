# MelBandRoformer/BSRoformer Tests

This directory contains the test suite for the MelBandRoformer/BSRoformer C++ implementation.

## Test Overview

| Test Name | Description | Requires External Data |
|-----------|-------------|------------------------|
| `test_audio` | Audio I/O functionality | ❌ |
| `test_component_stft` | STFT/ISTFT component verification | ❌ |
| `test_component_bandsplit` | BandSplit layer verification | ✅ |
| `test_component_layers` | Transformer layers verification | ✅ |
| `test_component_mask` | MaskEstimator verification | ✅ |
| `test_inference` | End-to-end inference verification | ✅ |
| `test_chunking_logic` | Chunking/overlap-add logic verification | ✅ |

## Quick Start

### 1. Build Tests

```powershell
# Configure with tests enabled
cmake -B build -DGGML_CUDA=ON -DMBR_BUILD_TESTS=ON

# Build
cmake --build build --config Release --parallel 14
```

### 2. Generate Test Data

First, clone the original PyTorch inference code repository:

```bash
git clone https://github.com/ZFTurbo/Music-Source-Separation-Training.git
```

Then use the script to generate test data:

```powershell
python scripts/generate_test_data.py `
    --model-repo "path/to/Music-Source-Separation-Training" `
    --audio "test_segment.wav" `
    --checkpoint "MelBandRoformer.ckpt" `
    --output "test_data"
```

> **Note:** 
> - `MelBandRoformer.ckpt` is the original PyTorch model weights file.
> - By default, the script extracts audio from **2.0s to 5.0s**. Use `--start` and `--end` to verify a different range.

### 3. Run Tests

Set environment variables and run:

```powershell
# Set environment variables
$env:MBR_MODEL_PATH = "path/to/model.gguf"
$env:MBR_TEST_DATA_DIR = "path/to/test_data"

# Run all tests
ctest --test-dir build -C Release

# Run specific test
ctest --test-dir build -C Release -R test_inference

# Show detailed output
ctest --test-dir build -C Release --output-on-failure
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MBR_MODEL_PATH` | Path to GGUF model file | `models/MelBandRoformer-228M-Vocals-v1-FP16.gguf` |
| `MBR_TEST_DATA_DIR` | Test data directory (containing `activations/` subdirectory) | `test_data` |
| `MBR_TEST_ATOL` | Absolute tolerance (optional) | `0.01` |
| `MBR_TEST_RTOL` | Relative tolerance (optional) | `0.01` |

## Test Data Structure

```
test_data/
├── chunk_in.npy              # Chunking test input
├── chunk_out.npy             # Chunking test output
└── activations/
    ├── input_audio.npy       # Input audio [1, 2, N]
    ├── output_audio.npy      # Output audio [1, 2, N]
    ├── band_split_in.npy     # BandSplit input
    ├── after_band_split.npy  # BandSplit output
    ├── before_mask_est.npy   # Transformer output
    └── mask_est0.npy         # MaskEstimator output
```

## Verification Standards

| Model Type | Expected Max Abs Diff | Expected Mean Abs Diff |
|------------|----------------------|------------------------|
| FP32 | < 1e-4 | < 1e-5 |
| FP16 | < 5e-4 | < 5e-5 |
| Q8_0 | < 5e-3 | < 5e-4 |
| Q5_x | < 2e-2 | < 3e-3 |
| Q4_x | < 5e-2 | < 5e-3 |

## Adding New Tests

1. Create `test_xxx.cpp` in `tests/` directory
2. Use utilities from `test_common.h`
3. Add to `tests/CMakeLists.txt`:
   ```cmake
   mbr_add_test(test_xxx)
   ```

## Troubleshooting

### Test fails: Model file not found

Ensure `MBR_MODEL_PATH` points to a valid `.gguf` file.

### Test fails: Test data not found

Ensure `MBR_TEST_DATA_DIR` points to a directory containing the `activations/` subdirectory.

### Numerical mismatch

For quantized models, relax the tolerance:
```powershell
$env:MBR_TEST_ATOL = "0.05"
```
