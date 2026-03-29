# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BSRoformer.cpp** is a high-performance C++ inference engine for BS Roformer and Mel-Band-Roformer audio source separation models. Built on the GGML tensor library, it extracts vocals or accompaniment from music with CPU/GPU acceleration support.

- **Language**: C++17
- **Build System**: CMake 3.17+
- **Dependencies**: GGML (tensor library), dr_wav (audio I/O)
- **Backends**: CPU, CUDA, Vulkan
- **Model Format**: GGUF with quantization support (FP32/FP16/Q8_0/Q4_0/Q5_0/Q5_1)

## Build Commands

### Basic Build

```bash
# CPU-only build
cmake -B build
cmake --build build --config Release --parallel

# CUDA acceleration (recommended)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release --parallel

# Vulkan acceleration
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release --parallel

# Build with tests enabled
cmake -B build -DGGML_CUDA=ON -DBSR_BUILD_TESTS=ON
cmake --build build --config Release --parallel
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `GGML_CUDA` | `ON` | Enable CUDA backend |
| `GGML_VULKAN` | `OFF` | Enable Vulkan backend |
| `BSR_BUILD_CLI` | `ON` | Build command-line tool |
| `BSR_BUILD_TESTS` | `OFF` | Build test suite |

### GGML Dependency

GGML is resolved automatically in this priority order:
1. Existing `ggml` target (from parent project)
2. Submodule at `ggml/`
3. Sibling directory at `../ggml`
4. Explicit path via `-DGGML_DIR=/path/to/ggml`

To set up GGML as submodule:
```bash
git submodule add https://github.com/ggerganov/ggml.git
git submodule update --init --recursive
```

See [GGML_DEPENDENCY.md](GGML_DEPENDENCY.md) for details.

## Testing

### Running Tests

```bash
# Set required environment variables
export BSR_MODEL_PATH="models/model.gguf"
export BSR_TEST_DATA_DIR="test_data"

# Run all tests
ctest --test-dir build -C Release

# Run specific test
ctest --test-dir build -C Release -R test_inference

# Verbose output
ctest --test-dir build -C Release --verbose
```

**Note**: Tests requiring external model/test data will be **skipped** (exit code 77) if files are missing, not failed.

### Test Categories

| Test | Requirements | Verification |
|------|--------------|--------------|
| `test_audio` | None | Audio read/write functionality |
| `test_component_stft` | None | STFT/ISTFT numerical precision |
| `test_component_bandsplit` | Model + test data | Band split layer |
| `test_component_layers` | Model + test data | Transformer layers |
| `test_component_mask` | Model + test data | Mask estimator |
| `test_inference` | Model + test data | End-to-end inference |
| `test_chunking_logic` | Model + test data | Chunking overlap-add logic |
| `test_inference_offset_stability` | Model | Offset stability verification |
| `test_streaming_overlap_add` | None | Streaming pipeline logic |

### Generating Test Data

Requires [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training):

```bash
git clone https://github.com/ZFTurbo/Music-Source-Separation-Training.git
cd Music-Source-Separation-Training
pip install -r requirements.txt
cd ..

python scripts/generate_test_data.py \
    --model-repo "Music-Source-Separation-Training" \
    --audio "test.wav" \
    --checkpoint "model.ckpt" \
    --output "test_data"
```

## CLI Usage

```bash
./bs_roformer-cli <model.gguf> <input.wav> <output.wav> [options]

Options:
  --chunk-size <N>   Chunk size in samples (default: model value, ~352800)
  --overlap <N>      Number of overlaps (default: model value, recommended: 2-4)
  --no-stream        Disable streaming I/O (debug only, uses more RAM)
  --no-pipeline      Disable pipelined inference (debug only)
  --help, -h         Show help message
```

**Examples:**
```bash
# Basic usage
./bs_roformer-cli model.gguf song.wav vocals.wav

# High quality mode (more overlaps reduce artifacts)
./bs_roformer-cli model.gguf song.wav vocals.wav --overlap 4

# Custom chunk size
./bs_roformer-cli model.gguf song.wav vocals.wav --chunk-size 352800 --overlap 2
```

**Requirements**: Input audio must be 44100 Hz (stereo or mono).

## Performance Tuning

Environment variables for advanced tuning:

| Variable | Default | Description |
|----------|---------|-------------|
| `BSR_STREAM_PIPELINE_DEPTH` | `3` (range: 1-8) | In-flight chunks in streaming pipeline. Higher = less GPU idle, more RAM |
| `BSR_GGML_GRAPH_CTX_MB` | `32` | GGML graph context size in MB. Increase if graph building fails |
| `BSR_STREAM_TIMING` | `0` | Set to `1` to print per-chunk timing (pre/inf/post stages) |

## Architecture

### Project Structure

```
BSRoformer.cpp/
├── include/bs_roformer/
│   ├── inference.h        # Public API: Inference engine
│   └── audio.h            # Public API: Audio I/O
├── src/
│   ├── model.h/cpp        # GGUF loading, graph building (internal)
│   ├── inference.cpp      # Core pipeline: STFT → Network → ISTFT
│   ├── stft.h             # Pure C++ STFT/ISTFT (Radix-2 FFT)
│   ├── audio.cpp          # WAV I/O (dr_wav wrapper)
│   └── utils.h/cpp        # NPY loading, tensor comparison
├── cli/main.cpp           # Command-line tool
├── tests/                 # Test suite
├── scripts/
│   ├── convert_to_gguf.py      # PyTorch → GGUF conversion
│   └── generate_test_data.py   # Test data generation
└── third_party/dr_libs/   # Audio library (header-only)
```

### Core Modules

**1. Model Loading (`model.h/cpp`)**

The `BSRoformer` class handles:
- GGUF weight loading and hyperparameter parsing
- Buffer generation (`freq_indices`, `num_bands_per_freq`)
- Computation graph building:
  - `BuildBandSplitGraph()` - Band split layer
  - `BuildTransformersGraph()` - Time-frequency Transformer stack
  - `BuildMaskEstimatorGraph()` - Mask estimation

**2. Inference Engine (`inference.cpp`)**

The `Inference` class implements the complete audio processing pipeline:

```
Input Audio → Chunking → STFT → Neural Network → Mask Application → ISTFT → Overlap-Add → Output
```

Key methods:
- `Process()` - Process complete audio with auto-chunking
- `ProcessChunk()` - Process single audio chunk
- `ComputeSTFT()` - Short-Time Fourier Transform
- `PostProcessAndISTFT()` - Mask application and inverse transform

**Pipeline Optimization**: CPU preprocessing and GPU inference run in parallel:
```
Chunk N:   [CPU Preprocess] → [GPU Inference] → [CPU Postprocess]
Chunk N+1:                   [CPU Preprocess] → [GPU Inference] → ...
                              ↑ Parallel execution reduces GPU idle time
```

**3. STFT Implementation (`stft.h`)**

Pure C++ implementation, numerically aligned with PyTorch `torch.stft/istft`:
- Radix-2 Cooley-Tukey FFT (O(N log N))
- Hann window with periodic mode
- Reflect-mode center padding
- OpenMP parallelization at frame level

**4. Audio I/O (`audio.h/cpp`)**

Lightweight wrapper around [dr_wav](https://github.com/mackron/dr_libs):
- Read: WAV file → float32 interleaved format
- Write: float32 interleaved format → WAV file
- Streaming support to avoid loading full tracks into RAM

## Model Conversion

Convert PyTorch weights to GGUF format:

```bash
# Install dependencies
pip install torch numpy pyyaml librosa einops gguf

# Convert model
python scripts/convert_to_gguf.py \
    --ckpt model.ckpt \
    --config config.yaml \
    --out model.gguf \
    --dtype q8_0

# For BS Roformer (usually auto-detected)
python scripts/convert_to_gguf.py ... --arch bs
```

### Quantization Types

| Type | Precision | Size | Use Case |
|------|-----------|------|----------|
| `fp32` | Highest | 100% | Debugging/baseline |
| `fp16` | High | 50% | High precision needs |
| `q8_0` | Good | 25% | **Recommended** (balance) |
| `q5_1` | Medium | 18% | Resource constrained |
| `q4_0` | Lower | 12.5% | Extreme compression |

**Note**: K-Quant types (Q4_K, Q5_K) are not supported by the conversion script.

## C++ API Example

```cpp
#include <atomic>
#include <bs_roformer/inference.h>
#include <bs_roformer/audio.h>

// Load audio
AudioBuffer input = AudioFile::Load("input.wav");

// Initialize inference engine
Inference engine("model.gguf");

// Get recommended parameters
int chunk_size = engine.GetDefaultChunkSize();   // e.g., 352800
int num_overlap = engine.GetDefaultNumOverlap(); // e.g., 2

// Run inference with progress callback
std::atomic<bool> should_cancel{false};
auto stems = engine.Process(input.data, chunk_size, num_overlap,
    [](float progress) {
        std::cout << "Progress: " << int(progress * 100) << "%" << std::endl;
    },
    [&should_cancel]() {
        return should_cancel.load();
    });

// Save result
AudioBuffer output{stems[0], 2, 44100, stems[0].size()};
AudioFile::Save("vocals.wav", output);
```

If `cancel_callback` returns `true`, `Process()` throws `std::runtime_error("Inference cancelled")`.

## Important Notes

- **Breaking Change**: Build/test prefixes were renamed from `MBR_*` to `BSR_*` with no compatibility aliases.
- **Input Requirements**: Audio must be 44100 Hz sample rate. Stereo or mono supported (mono auto-expanded).
- **Memory**: CLI uses streaming WAV I/O by default. Use `--no-stream` for legacy full-load behavior.
- **OpenMP**: Automatically detected and enabled if available (accelerates STFT).
