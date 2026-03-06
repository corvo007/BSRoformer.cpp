# BSRoformer.cpp

[中文](README.zh.md) | English

High-performance C++ inference implementation for the **BS Roformer** and **Mel-Band-Roformer** audio source separation model.

## 📖 Introduction

This project is a pure C++ inference engine for the **BS Roformer** and **Mel-Band-Roformer** audio source separation models, built on the [GGML](https://github.com/ggerganov/ggml) tensor library. It primarily used for extracting vocals or accompaniment from music.

### ✨ Key Features

- 🚀 **High-Performance Inference**: Supports CPU/GPU (CUDA, Vulkan) acceleration
- 🏗️ **Multi-Architecture**: Support for both **Mel-Band Roformer** and **BS Roformer**
- 📦 **GGUF Model Format**: Unified model file format for easy distribution
- 🎚️ **Multiple Quantization Support**: FP32/FP16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1
- 🔧 **Easy Deployment**: Only requires executable and GGML library
- 🎵 **Complete Audio Pipeline**: Built-in STFT/ISTFT and audio I/O
- ⚡ **Pipeline Optimization**: CPU preprocessing and GPU inference run in parallel

---

## 🚀 Quick Start

### Download

- **Pre-built Binaries**: Download executables for your platform from the [Releases](../../releases) page
- **GGUF Models**: Download pre-converted model files from [BSRoformer-GGUF](https://huggingface.co/chenmozhijin/BSRoformer-GGUF)

### Command Line Usage

```bash
./bs_roformer-cli <model.gguf> <input.wav> <output.wav> [options]

Options:
  --chunk-size <N>   Chunk size (in samples), defaults to model value
  --overlap <N>      Number of overlaps, defaults to model value
  --no-stream        Disable streaming I/O (debug only; uses more RAM)
  --no-io-threads    Streaming I/O without reader/writer threads (debug only)
  --no-pipeline      Disable pipelined streaming inference (debug only)
  --segment-minutes [N] Enable multiprocess segmentation for long audio (default N=30)
  --segment-overlap-seconds <N> Overlap duration for segment crossfade (default: 10)
  --segment-keep-temp Keep temporary segment outputs (debug only)
  --no-segment       Disable multiprocess segmentation (debug only)
  --pipeline-depth <N>  Streaming pipeline depth (1-8, default: 2)
  --cuda-pinned-staging Enable CUDA pinned staging (default: off)
  --no-progress      Disable progress bar output
  --help, -h         Show help message
```

**Parameter Description:**

| Parameter | Description |
|-----------|-------------|
| `--chunk-size` | Number of audio samples to process at once. Larger values require more VRAM but may improve processing efficiency. Default is typically 352800 (~8 seconds @44100Hz). |
| `--overlap` | Number of overlaps between chunks. Increasing this value can **improve output quality** as it helps reduce artifacts when reassembling chunks, but will increase inference time. Recommended value is 2-4. |

**Examples:**

```bash
# Basic usage (using model defaults)
./bs_roformer-cli model.gguf song.wav vocals.wav

# Custom chunking parameters
./bs_roformer-cli model.gguf song.wav vocals.wav --chunk-size 352800 --overlap 2

# High quality mode (increase overlap to reduce artifacts)
./bs_roformer-cli model.gguf song.wav vocals.wav --overlap 4
```

> **Note**: Input audio must be **44100 Hz**. Stereo or mono is supported (auto-expanded).
>
> **Memory**: The CLI uses **streaming WAV read/write by default** to avoid loading the full track into RAM. Use `--no-stream` to fall back to the legacy full-load path.
>
> **Very long audio**: By default, inputs longer than 30 minutes are processed via **multiprocess segmentation** (30-minute segments, with a crossfade overlap) to cap CUDA host RAM growth over time. Use `--no-segment` to force single-process, or tune with `--segment-minutes N`.

### Performance Tuning (Advanced)

You can tune performance and memory usage via **command-line arguments** (recommended) or environment variables:

**Command-line arguments** (override environment variables):
- `--pipeline-depth <N>` (default `2`, range `1-8`): Number of in-flight chunks in the streaming pipeline. Higher values reduce GPU idle time but increase RAM usage.
- `--cuda-pinned-staging`: Enable pinned host staging buffers for CUDA copies. Can improve throughput but increases locked memory usage.

**Environment variables** (fallback if CLI args not set):
- `BSR_STREAM_PIPELINE_DEPTH` (default `2`, range `1-8`): Same as `--pipeline-depth`
- `BSR_CUDA_PINNED_STAGING` (default `0`): Set to `1` to enable, same as `--cuda-pinned-staging`

**Examples:**
```bash
# Reduce memory usage (lower pipeline depth)
./bs_roformer-cli model.gguf input.wav output.wav --pipeline-depth 1

# Maximize throughput (enable pinned staging)
./bs_roformer-cli model.gguf input.wav output.wav --cuda-pinned-staging
```
- `BSR_GGML_GRAPH_CTX_MB` (default `32`): GGML graph context size in MB. Increase if graph building fails for a specific model/chunk size.
- `BSR_STREAM_TIMING` (default `0`): set to `1` to print per-chunk timing for `pre/inf/post` stages (useful for GPU bubble analysis).

---

#### Recommended Settings (CUDA)

Based on internal benchmarks on Windows 11 + RTX 4070 SUPER (`becruily_deux-Q8_0.gguf`):

- **Best balance**: keep defaults (`BSR_STREAM_PIPELINE_DEPTH=2`).
- **Lower peak RAM**: set `BSR_STREAM_PIPELINE_DEPTH=1` (slower, but noticeably less host memory).
- **Depth > 2**: typically yields very small speed gains, but increases host RAM (each extra in-flight chunk is large).
- **Pinned staging**: default is off (`BSR_CUDA_PINNED_STAGING=0`). If you prioritize throughput and have spare RAM, set `BSR_CUDA_PINNED_STAGING=1`.

Measured impact of `BSR_STREAM_PIPELINE_DEPTH` on CUDA (Optimized) + `becruily_deux`:

| Depth | 5min time (s) | 5min peak WS (MB) | 24min time (s) | 24min peak WS (MB) |
|------:|--------------:|------------------:|---------------:|-------------------:|
| 1 | 36.61 | 960.8  | 162.75 | 1526.9 |
| 2 | 31.91 | 1112.4 | 140.71 | 1678.6 |
| 3 | 31.89 | 1263.8 | 139.90 | 1828.5 |
| 4 | 31.89 | 1410.7 | 139.98 | 1976.0 |

For more details, see `docs/benchmark_report.md`.

## 🔧 Building from Source

### Prerequisites

- CMake >= 3.17
- C++17 compatible compiler (MSVC 2019+, GCC 9+, Clang 10+)
- GGML source code (submodule or local directory)

### Getting GGML Dependency

The project supports multiple ways to obtain GGML:

```bash
# Option 1: Git Submodule (Recommended)
git submodule add https://github.com/ggerganov/ggml.git
git submodule update --init --recursive

# Option 2: Sibling Directory
cd ..
git clone https://github.com/ggerganov/ggml.git

# Option 3: Explicit Path
cmake -B build -DGGML_DIR=/path/to/ggml
```

See [GGML_DEPENDENCY.md](GGML_DEPENDENCY.md) for details.

### Build Commands

```bash
# CPU Build
cmake -B build
cmake --build build --config Release --parallel

# CUDA Acceleration (Recommended)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release --parallel

# Enable Tests
cmake -B build -DGGML_CUDA=ON -DBSR_BUILD_TESTS=ON
cmake --build build --config Release --parallel
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `GGML_CUDA` | `ON` | Enable CUDA backend |
| `BSR_BUILD_CLI` | `ON` | Build command line tool |
| `BSR_BUILD_TESTS` | `OFF` | Build test suite |

> **Breaking Change:** build/test prefixes were renamed from `MBR_*` to `BSR_*` with no compatibility aliases.

---

## 📦 Model Conversion

If you need to convert models yourself, use `convert_to_gguf.py` to convert PyTorch weights to GGUF format.

**Install Dependencies:**

```bash
pip install torch numpy pyyaml librosa einops gguf
```

**Conversion Command:**

```bash
python scripts/convert_to_gguf.py \
    --ckpt model.ckpt \
    --config config.yaml \
    --out model.gguf \
    --dtype q8_0

# For BS Roformer (optional, usually auto-detected)
python scripts/convert_to_gguf.py ... --arch bs
```

### Supported Quantization Types

| Type | Precision | Size | Recommended Use |
|------|-----------|------|-----------------|
| `fp32` | Highest | 100% | Debugging/Baseline |
| `fp16` | High | 50% | High precision needs |
| `q8_0` | Good | 25% | **Recommended** (balance of precision and performance) |
| `q5_1` | Medium | 18% | Resource constrained |
| `q4_0` | Lower | 12.5% | Extreme compression |

> **Note**: The conversion script currently does not support K-Quant types (Q4_K, Q5_K, etc.). This is mainly because the gguf-py library has not yet implemented K-Quant quantization (only supports reading/dequantization), and most models do not meet the requirement that dim must be divisible by 256.

---

## 💻 C++ API

```cpp
#include <atomic>
#include <bs_roformer/inference.h>
#include <bs_roformer/audio.h>

// 1. Load audio file
AudioBuffer input = AudioFile::Load("input.wav");

// 2. Initialize inference engine
Inference engine("model.gguf");

// 3. Get model's recommended inference parameters
int chunk_size = engine.GetDefaultChunkSize();   // e.g., 352800
int num_overlap = engine.GetDefaultNumOverlap(); // e.g., 2

// 4. Run inference (with progress + cancel callback)
std::atomic<bool> should_cancel{false};
auto stems = engine.Process(input.data, chunk_size, num_overlap,
    [](float progress) {
        std::cout << "Progress: " << int(progress * 100) << "%" << std::endl;
    },
    [&should_cancel]() {
        return should_cancel.load();
    });

// 5. Save result
AudioBuffer output{stems[0], 2, 44100, stems[0].size()};
AudioFile::Save("vocals.wav", output);
```

If `cancel_callback` returns `true`, `Process()` throws `std::runtime_error("Inference cancelled")`.

---

## 🏗️ Project Architecture

```
BSRoformer.cpp/
├── include/
│   └── bs_roformer/
│       ├── inference.h        # Inference Engine API
│       └── audio.h            # Audio I/O API
├── src/
│   ├── model.h/cpp            # Model weight loading & graph building (internal)
│   ├── inference.cpp          # Core inference logic (STFT → Network → ISTFT)
│   ├── stft.h                 # STFT/ISTFT implementation (Radix-2 FFT)
│   ├── audio.cpp              # Audio read/write implementation (dr_wav)
│   └── utils.h/cpp            # NPY loading, tensor comparison tools
├── third_party/
│   └── dr_libs/dr_wav.h       # dr_libs audio library
├── cli/
│   └── main.cpp               # Command line tool
├── scripts/
│   ├── convert_to_gguf.py      # PyTorch → GGUF conversion tool
│   ├── generate_test_data.py   # Test data generation script
│   └── generate_test_audio.py  # CI test audio generation (no external files needed)
├── tests/                     # Unit test suite
├── models/                    # Model file directory
└── CMakeLists.txt             # Build configuration
```

---

## 📐 Core Module Details

### 1. Model Loading (`model.h/cpp`)

The `BSRoformer` class is responsible for:

- **GGUF Weight Loading**: Parsing hyperparameters and tensors from file
- **Buffer Generation**: `freq_indices`, `num_bands_per_freq`, etc.
- **Computation Graph Building**:
  - `BuildBandSplitGraph()` - Band split layer
  - `BuildTransformersGraph()` - Time-frequency Transformer stacking
  - `BuildMaskEstimatorGraph()` - Mask estimator

### 2. Inference Engine (`inference.cpp`)

The `Inference` class implements the complete audio processing pipeline:

```
Input Audio → Chunking → STFT → Neural Network → Mask Application → ISTFT → Overlap-Add → Output
```

**Key Methods**:

| Method | Function |
|--------|----------|
| `Process()` | Process complete audio (auto-chunking) |
| `ProcessChunk()` | Process a single audio chunk |
| `ComputeSTFT()` | Short-Time Fourier Transform |
| `PostProcessAndISTFT()` | Mask application and inverse transform |

**Pipeline Optimization**:

```
Chunk N:   [CPU Preprocess] → [GPU Inference] → [CPU Postprocess]
Chunk N+1:                   [CPU Preprocess] → [GPU Inference] → [CPU Postprocess]
                              ↑ Parallel execution
```

### 3. STFT Implementation (`stft.h`)

Pure C++ implementation, numerically aligned with PyTorch `torch.stft/istft`:

- **Radix-2 Cooley-Tukey FFT**: Efficient O(N log N) implementation
- **Hann Window**: Periodic window function
- **Center Padding**: Reflect mode padding
- **OpenMP Parallelization**: Frame-level parallel acceleration

### 4. Audio I/O (`audio.h/cpp`)

Lightweight audio processing based on [dr_libs](https://github.com/mackron/dr_libs):

- Read: WAV file → `float32` interleaved format
- Write: `float32` interleaved format → WAV file

---

## 🧪 Testing

### Running Tests

```bash
# Set environment variables
$env:BSR_MODEL_PATH = "models/model.gguf"
$env:BSR_TEST_DATA_DIR = "test_data"

# Run all tests
ctest --test-dir build -C Release

# Run specific test
ctest --test-dir build -C Release -R test_inference
```

> Note: Tests that require an external model or `test_data/` will be **skipped** automatically if the required files are missing.

### Test Suite

| Test File | Verification Content |
|-----------|---------------------|
| `test_audio` | Audio read/write functionality |
| `test_component_stft` | STFT/ISTFT numerical precision |
| `test_component_bandsplit` | Band split layer |
| `test_component_layers` | Transformer layers |
| `test_component_mask` | Mask estimator |
| `test_inference` | End-to-end inference |
| `test_chunking_logic` | Chunking overlap-add logic |

### Generating Test Data

First clone [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) and install its dependencies:

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

---

## Acknowledgements

- [ggerganov/ggml](https://github.com/ggerganov/ggml) - Efficient tensor library
- [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) - PyTorch reference implementation
- [dr_libs](https://github.com/mackron/dr_libs) - Lightweight audio library
