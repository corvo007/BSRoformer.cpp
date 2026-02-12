# MelBandRoformer.cpp

[中文](README.zh.md) | English

High-performance C++ inference implementation for the Mel-Band-Roformer audio source separation model.

## 📖 Introduction

This project is a pure C++ inference engine for the **Mel-Band-Roformer** and **BS Roformer** audio source separation models, built on the [GGML](https://github.com/ggerganov/ggml) tensor library. It primarily used for extracting vocals or accompaniment from music.

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
- **GGUF Models**: Download pre-converted model files from [MelBandRoformer-GGUF](https://huggingface.co/chenmozhijin/MelBandRoformer-GGUF)

### Command Line Usage

```bash
./mel_band_roformer-cli <model.gguf> <input.wav> <output.wav> [options]

Options:
  --chunk-size <N>   Chunk size (in samples), defaults to model value
  --overlap <N>      Number of overlaps, defaults to model value
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
./mel_band_roformer-cli model.gguf song.wav vocals.wav

# Custom chunking parameters
./mel_band_roformer-cli model.gguf song.wav vocals.wav --chunk-size 352800 --overlap 2

# High quality mode (increase overlap to reduce artifacts)
./mel_band_roformer-cli model.gguf song.wav vocals.wav --overlap 4
```

> **Note**: Input audio must be **44100 Hz**. Stereo or mono is supported (auto-expanded).

---

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
cmake -B build -DGGML_CUDA=ON -DMBR_BUILD_TESTS=ON
cmake --build build --config Release --parallel
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `GGML_CUDA` | `ON` | Enable CUDA backend |
| `MBR_BUILD_CLI` | `ON` | Build command line tool |
| `MBR_BUILD_TESTS` | `OFF` | Build test suite |

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
#include <mel_band_roformer/inference.h>
#include <mel_band_roformer/audio.h>

// 1. Load audio file
AudioBuffer input = AudioFile::Load("input.wav");

// 2. Initialize inference engine
Inference engine("model.gguf");

// 3. Get model's recommended inference parameters
int chunk_size = engine.GetDefaultChunkSize();   // e.g., 352800
int num_overlap = engine.GetDefaultNumOverlap(); // e.g., 2

// 4. Run inference (with progress callback)
auto stems = engine.Process(input.data, chunk_size, num_overlap,
    [](float progress) {
        std::cout << "Progress: " << int(progress * 100) << "%" << std::endl;
    });

// 5. Save result
AudioBuffer output{stems[0], 2, 44100, stems[0].size()};
AudioFile::Save("vocals.wav", output);
```

---

## 🏗️ Project Architecture

```
MelBandRoformer.cpp/
├── include/
│   └── mel_band_roformer/
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

The `MelBandRoformer` class is responsible for:

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
$env:MBR_MODEL_PATH = "models/model.gguf"
$env:MBR_TEST_DATA_DIR = "test_data"

# Run all tests
ctest --test-dir build -C Release

# Run specific test
ctest --test-dir build -C Release -R test_inference
```

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
