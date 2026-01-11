# MelBandRoformer.cpp

Mel-Band-Roformer 音频源分离模型的高性能 C++ 推理实现。

## 📖 简介

本项目是 Mel-Band-Roformer 音频源分离模型的纯 C++ 推理引擎，基于 [GGML](https://github.com/ggerganov/ggml) 张量库构建。理论上支持大部分 Mel-Band-Roformer 模型，主要用于从音乐中提取人声或伴奏。

### ✨ 主要特性

- 🚀 **高性能推理**：支持 CPU/GPU (CUDA、Vulkan) 加速
- 📦 **GGUF 模型格式**：统一的模型文件格式，易于分发
- 🎚️ **多种量化支持**：FP32/FP16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1
- 🔧 **易于部署**：仅需可执行文件和 GGML 库
- 🎵 **完整音频流程**：内置 STFT/ISTFT 和音频 I/O
- ⚡ **流水线优化**：CPU 预处理与 GPU 推理并行执行

---

## 🚀 快速开始

### 下载

- **预构建程序**：在 [Releases](../../releases) 页面下载对应平台的可执行文件
- **GGUF 模型**：在 [MelBandRoformer-GGUF](https://huggingface.co/MelBandRoformer-GGUF) 下载预转换的模型文件

### 命令行使用

```bash
./mel_band_roformer-cli <模型.gguf> <输入.wav> <输出.wav> [选项]

选项:
  --chunk-size <N>   分块大小（采样点数），默认从模型读取
  --overlap <N>      重叠数量，默认从模型读取
  --help, -h         显示帮助信息
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--chunk-size` | 每次处理的音频采样点数。较大的值需要更多显存，但可能提高处理效率。默认值通常为 352800（约 8 秒 @44100Hz）。 |
| `--overlap` | 分块之间的重叠数量。增加此值可以**提高输出质量**，因为它有助于减少重新组合数据块时产生的伪影（artifacts），但会延长推理时间。建议值为 2-4。 |

**示例：**

```bash
# 基本用法（使用模型默认参数）
./mel_band_roformer-cli model.gguf song.wav vocals.wav

# 自定义分块参数
./mel_band_roformer-cli model.gguf song.wav vocals.wav --chunk-size 352800 --overlap 2

# 高质量模式（增加重叠，减少伪影）
./mel_band_roformer-cli model.gguf song.wav vocals.wav --overlap 4
```

> **注意**：输入音频必须为 **44100 Hz**，支持立体声或单声道（自动扩展）。

---

## 🔧 从源码构建

### 前置条件

- CMake >= 3.17
- C++17 兼容编译器（MSVC 2019+, GCC 9+, Clang 10+）
- GGML 源码（submodule 或本地目录）

### 获取 GGML 依赖

项目支持多种 GGML 获取方式：

```bash
# 方式一：Git Submodule（推荐）
git submodule add https://github.com/ggerganov/ggml.git
git submodule update --init --recursive

# 方式二：兄弟目录
cd ..
git clone https://github.com/ggerganov/ggml.git

# 方式三：显式指定路径
cmake -B build -DGGML_DIR=/path/to/ggml
```

详见 [GGML_DEPENDENCY.md](GGML_DEPENDENCY.md)。

### 编译命令

```bash
# CPU 构建
cmake -B build
cmake --build build --config Release --parallel

# CUDA 加速（推荐）
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release --parallel

# 启用测试
cmake -B build -DGGML_CUDA=ON -DMBR_BUILD_TESTS=ON
cmake --build build --config Release --parallel
```

### CMake 选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `GGML_CUDA` | `ON` | 启用 CUDA 后端 |
| `MBR_BUILD_CLI` | `ON` | 构建命令行工具 |
| `MBR_BUILD_TESTS` | `OFF` | 构建测试套件 |

---

## 📦 模型转换

如果需要自行转换模型，可使用 `convert_to_gguf.py` 将 PyTorch 权重转换为 GGUF 格式。

**依赖安装：**

```bash
pip install torch numpy pyyaml librosa einops gguf
```

**转换命令：**

```bash
python scripts/convert_to_gguf.py \
    --ckpt model.ckpt \
    --config config.yaml \
    --out model.gguf \
    --dtype q8_0
```

### 支持的量化类型

| 类型 | 精度 | 体积 | 推荐场景 |
|------|------|------|----------|
| `fp32` | 最高 | 100% | 调试/基准 |
| `fp16` | 高 | 50% | 高精度需求 |
| `q8_0` | 较高 | 25% | **推荐**（平衡精度与性能） |
| `q5_1` | 中等 | 18% | 资源受限 |
| `q4_0` | 较低 | 12.5% | 极限压缩 |

> **注意**：目前转换脚本不支持 K-Quant 类型 (Q4_K, Q5_K 等)，主要原因是 gguf-py 库尚未实现 K-Quant 的量化功能（仅支持读取/反量化），并且大部分模型不满足dim能被256整除的条件。

---

## 💻 C++ API

```cpp
#include <mel_band_roformer/inference.h>
#include <mel_band_roformer/audio.h>

// 1. 加载音频文件
AudioBuffer input = AudioFile::Load("input.wav");

// 2. 初始化推理引擎
Inference engine("model.gguf");

// 3. 获取模型推荐的推理参数
int chunk_size = engine.GetDefaultChunkSize();   // 如 352800
int num_overlap = engine.GetDefaultNumOverlap(); // 如 2

// 4. 执行推理（带进度回调）
auto stems = engine.Process(input.data, chunk_size, num_overlap,
    [](float progress) {
        std::cout << "Progress: " << int(progress * 100) << "%" << std::endl;
    });

// 5. 保存结果
AudioBuffer output{stems[0], 2, 44100, stems[0].size()};
AudioFile::Save("vocals.wav", output);
```

---

## 🏗️ 项目架构

```
MelBandRoformer.cpp/
├── include/
│   └── mel_band_roformer/
│       ├── inference.h        # 推理引擎 API
│       └── audio.h            # 音频 I/O API
├── src/
│   ├── model.h/cpp            # 模型权重加载与图构建（内部）
│   ├── inference.cpp          # 核心推理逻辑（STFT → 网络 → ISTFT）
│   ├── stft.h                 # STFT/ISTFT 实现（Radix-2 FFT）
│   ├── audio.cpp              # 音频读写实现（dr_wav）
│   └── utils.h/cpp            # NPY 加载、张量对比工具
├── third_party/
│   └── dr_libs/dr_wav.h       # dr_libs 音频库
├── cli/
│   └── main.cpp               # 命令行工具
├── scripts/
│   ├── convert_to_gguf.py      # PyTorch → GGUF 转换工具
│   ├── generate_test_data.py   # 测试数据生成脚本
│   └── generate_test_audio.py  # CI 测试音频生成（无需外部文件）
├── tests/                     # 单元测试套件
├── models/                    # 模型文件目录
└── CMakeLists.txt             # 构建配置
```

---

## 📐 核心模块详解

### 1. 模型加载 (`model.h/cpp`)

`MelBandRoformer` 类负责：

- **GGUF 权重加载**：从文件解析超参数和张量
- **缓冲区生成**：`freq_indices`、`num_bands_per_freq` 等
- **计算图构建**：
  - `BuildBandSplitGraph()` - 频带分割层
  - `BuildTransformersGraph()` - 时频 Transformer 堆叠
  - `BuildMaskEstimatorGraph()` - 掩码估计器

### 2. 推理引擎 (`inference.cpp`)

`Inference` 类实现完整的音频处理流程：

```
输入音频 → 分块(Chunking) → STFT → 神经网络 → 掩码应用 → ISTFT → 重叠相加 → 输出
```

**关键方法**：

| 方法 | 功能 |
|------|------|
| `Process()` | 处理完整音频（自动分块） |
| `ProcessChunk()` | 处理单个音频块 |
| `ComputeSTFT()` | 短时傅里叶变换 |
| `PostProcessAndISTFT()` | 掩码应用与逆变换 |

**流水线优化**：

```
Chunk N:   [CPU预处理] → [GPU推理] → [CPU后处理]
Chunk N+1:              [CPU预处理] → [GPU推理] → [CPU后处理]
                         ↑ 并行执行
```

### 3. STFT 实现 (`stft.h`)

纯 C++ 实现，与 PyTorch `torch.stft/istft` 数值对齐：

- **Radix-2 Cooley-Tukey FFT**：高效 O(N log N) 实现
- **Hann 窗口**：周期性窗函数
- **中心填充**：反射模式 (reflect padding)
- **OpenMP 并行**：帧级并行加速

### 4. 音频 I/O (`audio.h/cpp`)

基于 [dr_libs](https://github.com/mackron/dr_libs) 的轻量级音频处理：

- 读取：WAV 文件 → `float32` 交错格式
- 写入：`float32` 交错格式 → WAV 文件

---

## 🧪 测试

### 运行测试

```bash
# 设置环境变量
$env:MBR_MODEL_PATH = "models/model.gguf"
$env:MBR_TEST_DATA_DIR = "test_data"

# 运行所有测试
ctest --test-dir build -C Release

# 运行特定测试
ctest --test-dir build -C Release -R test_inference
```

### 测试套件

| 测试文件 | 验证内容 |
|----------|----------|
| `test_audio` | 音频读写功能 |
| `test_component_stft` | STFT/ISTFT 数值精度 |
| `test_component_bandsplit` | 频带分割层 |
| `test_component_layers` | Transformer 层 |
| `test_component_mask` | 掩码估计器 |
| `test_inference` | 端到端推理 |
| `test_chunking_logic` | 分块重叠相加逻辑 |

### 生成测试数据

需要先克隆 [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) 并安装其依赖：

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

## 致谢

- [ggerganov/ggml](https://github.com/ggerganov/ggml) - 高效张量库
- [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) - PyTorch 参考实现
- [dr_libs](https://github.com/mackron/dr_libs) - 轻量级音频库
