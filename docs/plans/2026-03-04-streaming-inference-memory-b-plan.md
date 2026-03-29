# Streaming Inference + Streaming WAV I/O (B 方案) Implementation Plan

## Status Update (2026-03-05)

- ✅ Streaming CLI 默认开启（`--no-stream` 可回退到整段加载，便于 debug）
- ✅ 恒定内存 overlap-add streamer + 3-stage pipeline（`BSR_STREAM_PIPELINE_DEPTH=1..8`）
- ✅ 修复 `voc_fv6`（`num_overlap=1`）多 chunk 变得极其安静的问题：每次 inference 都重新上传 RoPE position tensors（`src/inference.cpp:486`）
- ✅ 降低 GPU bubble（同步开销）：改用 `ggml_backend_tensor_{set,get}_async` + `ggml_backend_graph_compute_async`，每个 chunk 只 `ggml_backend_synchronize()` 一次（`src/inference.cpp:486`）
- ✅ 分阶段计时：`BSR_STREAM_TIMING=1` 打印 `pre/inf/post`；`BSR_STREAM_TIMING=2` 额外打印 `inf_detail(h2d/compute/d2h)`（`src/inference.cpp:98` / `src/inference.cpp:561`）
- ✅ 分阶段内存日志：`BSR_STREAM_MEM=1..2` 打印 `ws_mb/priv_mb`；CUDA 下额外打印 `gpu_used_mb/gpu_free_mb`（`src/inference.cpp`，`StreamImpl::MaybeLogMem`）
- ✅ CUDA graphs（可选但推荐）：编译时开启 `GGML_CUDA_GRAPHS=ON`，运行时默认启用；可用 `GGML_CUDA_DISABLE_GRAPHS=1` 关闭
- ✅ 回归测试：`tests/test_inference_offset_stability.cpp`（偏移稳定性）+ `tests/test_inference_repeatability.cpp`（重复性）
- ✅ 本机脚本构建 + 测试已跑通：`.\build-cuda.ps1 -Tests -RunTests`（ctest 12/12 passed，部分测试因外部资源 Skip；`test_streaming_overlap_add` PASS）
- ✅ 端到端音量验证（CUDA）：`voc_fv6` 输出 `max_volume=-0.6 dB`；`becruily_deux` 两 stem 输出正常（约 `-1.2 dB / 0.0 dB`）
- 🔎 现象更新（与 benchmark_report 对齐）：Streaming 的“算法层”内存已恒定（in-flight buffers 容量稳定），但 CUDA backend 下进程 `Working Set / Private Bytes` 仍可能随 chunk 逐步上升；新增 `gpu_used_mb` 日志后确认 VRAM 使用量全程恒定，因此这部分增长来自 **host 侧 runtime/driver/cache**，不是我们仍在偷偷累积整段音频数据。

### Build / Test (PowerShell + VS Build Tools + Ninja)

```powershell
chcp 65001
.\build-cuda.ps1 -Tests -RunTests
# 如需强制开启 CUDA graphs（若你的 CMake 默认没开）：
# .\build-cuda.ps1 -CudaGraphs -Tests -RunTests
```

### Run model-required tests (needs model file)

在 VS Developer Prompt（或 `vcvarsall.bat` 环境）中：

```bat
set BSR_MODEL_PATH=D:\Download\voc_fv6-Q8_0.gguf
set BSR_TEST_AUDIO_PATH=D:\onedrive\codelab\Gemini-Subtitle-Pro\MelBandRoformer.cpp\test_song_64s_8s.wav
ctest --test-dir build-cuda -C Release -R test_inference_repeatability --output-on-failure
ctest --test-dir build-cuda -C Release -R test_inference_offset_stability --output-on-failure
```

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce peak RAM for long-audio separation (e.g., 25 minutes @ 44.1kHz stereo) from ~3GB to **<500MB** by adding a **streaming inference API** and a **streaming CLI path** (incremental WAV read/write). Keep the existing `Inference::Process()` API working for backward compatibility.

**Architecture (high-level):**
- Replace O(N) “full-length overlap-add buffers” with a **constant-memory overlap-add streamer** that only keeps a sliding window of length `chunk_size` and returns `step = chunk_size / num_overlap` samples as soon as they are final.
- Add `Inference::{CreateStream, ProcessStream, FinalizeStream}` that wraps the existing `ProcessChunk()` as the model callback.
- Update `bs_roformer-cli` to avoid `AudioFile::Load()` and instead **read WAV frames incrementally** and **write stems incrementally**.
- Make GGML graph context size configurable with safe fallback (repo currently uses hardcoded `32MB` at `src/inference.cpp:84`).

**Tech Stack:** C++17, ggml, dr_wav, CMake/CTest.

---

## Background / Current Root Cause (grounded in code)

Current peak RAM is dominated by **full-length allocations** in overlap-add (linear in track duration):
- `src/inference.cpp:485` `padded_input` (full track)
- `src/inference.cpp:515` `counter` (full track)
- `src/inference.cpp:563` `result` (full track × stems)
  - Note: repo already removed the extra `final_output_stems` copy by normalizing + cropping in-place, but `result/counter/padded_input` are still O(N).

GGML graph context is currently **hardcoded** (already reduced, but not configurable):
- `src/inference.cpp:84` `size_t mem_size = 32ull * 1024 * 1024;`

CLI loads full WAV into memory:
- `cli/main.cpp:91` `AudioFile::Load()`
- `src/audio.cpp` previously used `drwav_open_file_and_read_pcm_frames_f32(...)` + copy; repo already switched to `drwav_init_file` + read directly into `std::vector<float>`, but it’s still full-file load (O(N) RAM).

## Already Implemented (repo state as of 2026-03-04)

- ✅ Reduce GGML graph ctx from 1GB → 32MB by default, and make it configurable via `BSR_GGML_GRAPH_CTX_MB` with safe fallback (`src/inference.cpp`)
- ✅ Avoid extra output allocation by in-place normalize + crop in `ProcessOverlapAddPipelined` (`src/inference.cpp`, normalize/crop section)
- ✅ Remove `AudioFile::Load()` temporary buffer copy (direct read into `std::vector<float>`) (`src/audio.cpp`)
- ✅ Add constant-memory overlap-add streamer + streaming inference API (`include/bs_roformer/inference.h`, `src/inference.cpp`)
- ✅ Add regression test: streaming overlap-add matches batch (`tests/test_streaming_overlap_add.cpp`, `tests/CMakeLists.txt`)
- ✅ CLI default uses streaming WAV I/O (no full-file buffers). Debug fallback via `--no-stream` (`cli/main.cpp`)
- ✅ GGML graph ctx is configurable + safer: `BSR_GGML_GRAPH_CTX_MB` with candidate fallback and an external backing buffer to avoid ggml internal malloc/assert (`src/inference.cpp`, `include/bs_roformer/inference.h`)
- ✅ Pipelined streaming inference (3-stage CPU/GPU pipeline) is available and default ON (`src/inference.cpp`, `Inference::CreateStream(..., pipelined=true)`), with `BSR_STREAM_PIPELINE_DEPTH` tuning
- ✅ Reduce GPU bubble further by optimizing CPU hot paths:
  - STFT/ISTFT: reuse buffers + avoid per-frame FFT instance lookup/lock (`src/stft.h`)
  - Postprocess: remove large intermediate `masks` / `stft_output_masked` buffers (`src/inference.cpp`, `PostProcessAndISTFT`)

## Current Execution Status (as of 2026-03-04)

- Completed: Task 1–7 (+ CPU hot-path optimizations to reduce GPU bubble)
- Remaining: Manual 25-min run (peak RAM) + GPU profiling/tuning (pipeline depth, backend, chunk size)

---

## Acceptance Criteria

**Functional:**
1. `bs_roformer-cli` can process a 25-minute 44.1kHz stereo WAV without OOM and with peak RAM **<500MB** (target **<350MB**).
2. Output audio must match current behavior for chunking/windowing (within float tolerance):
   - For identity model callback, streaming overlap-add output must match `Inference::ProcessOverlapAdd(...)` exactly (or within `1e-6`).
3. Backward compatibility:
   - Existing `Inference::Process()` and `Inference::ProcessOverlapAdd(...)` remain available and unchanged in signature.

**Testing:**
1. All existing tests pass: `ctest --test-dir build -C Release`.
2. New tests for streaming overlap-add pass without requiring a model file.

---

## Non-Goals (for this plan)
- Streaming STFT/ISTFT refactor (nice-to-have; defer unless needed after streaming overlap-add + streaming CLI).
- GGML multi-graph caching for performance (likely low ROI unless `n_frames` varies frequently).

---

## Task 1: Add failing unit test for streaming overlap-add

**Why first:** Locks behavior to match existing `ProcessOverlapAdd` exactly, without ggml/model dependency.

**Files:**
- Create: `tests/test_streaming_overlap_add.cpp`
- Modify: `tests/CMakeLists.txt`

### Step 1: Write failing test (API doesn’t exist yet)

Create `tests/test_streaming_overlap_add.cpp`:

```cpp
#include "test_common.h"
#include "bs_roformer/inference.h"
#include <random>
#include <numeric>

static std::vector<float> MakeRandomInterleavedStereo(int n_samples_per_ch, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> out(n_samples_per_ch * 2);
    for (int i = 0; i < n_samples_per_ch; ++i) {
        out[i * 2 + 0] = dist(rng);
        out[i * 2 + 1] = dist(rng);
    }
    return out;
}

static void AppendStems(std::vector<std::vector<float>>& dst, const std::vector<std::vector<float>>& src) {
    if (src.empty()) return;
    if (dst.empty()) dst.resize(src.size());
    for (size_t s = 0; s < src.size(); ++s) {
        dst[s].insert(dst[s].end(), src[s].begin(), src[s].end());
    }
}

int main() {
    std::cout << "Test: Streaming Overlap-Add matches ProcessOverlapAdd (identity model)\n";

    const int chunk_size = 1000;   // divisible by 10
    const int num_overlap = 2;     // step=500, border=500, 2*border=1000

    auto identity = [](const std::vector<float>& chunk) {
        return std::vector<std::vector<float>>{chunk};
    };

    // Cover both do_pad=false (<= 2*border) and do_pad=true (> 2*border)
    const int test_lengths[] = { 200, 800, 1000, 1001, 1200, 4096 };
    const int push_sizes[] = { 7, 123, 499, 777 }; // irregular push sizes to stress buffering

    for (int n_samples : test_lengths) {
        auto input = MakeRandomInterleavedStereo(n_samples, /*seed=*/(uint32_t)n_samples);
        auto expected = Inference::ProcessOverlapAdd(input, chunk_size, num_overlap, identity);

        Inference::OverlapAddStreamer streamer(chunk_size, num_overlap, /*num_stems=*/1, identity);

        std::vector<std::vector<float>> actual;
        int cursor = 0;
        int push_idx = 0;
        while (cursor < n_samples) {
            int push = push_sizes[push_idx++ % (sizeof(push_sizes)/sizeof(push_sizes[0]))];
            int take = std::min(push, n_samples - cursor);

            // take samples per channel => interleaved float count is take*2
            std::vector<float> slice(input.begin() + cursor * 2, input.begin() + (cursor + take) * 2);
            auto out = streamer.Push(slice);
            AppendStems(actual, out);

            cursor += take;
        }

        auto tail = streamer.Finalize();
        AppendStems(actual, tail);

        bool pass = CompareAndReport("stream-vs-batch",
            expected[0].data(), expected[0].size(),
            actual[0].data(), actual[0].size(),
            1e-6f, 1e-6f);
        if (!pass) {
            LOG_FAIL();
            return 1;
        }
    }

    LOG_PASS();
    return 0;
}
```

### Step 2: Register test in CMake

Modify `tests/CMakeLists.txt` by adding:

```cmake
bsr_add_test(test_streaming_overlap_add)
```

### Step 3: Verify it fails (compile error expected)

Run:

```bash
cmake -B build -DBSR_BUILD_TESTS=ON
cmake --build build --config Release --parallel
ctest --test-dir build -C Release -R test_streaming_overlap_add --output-on-failure
```

Expected: compile fails because `Inference::OverlapAddStreamer` is not defined.

### Step 4: Commit (optional)

```bash
git add tests/test_streaming_overlap_add.cpp tests/CMakeLists.txt
git commit -m "test: add streaming overlap-add regression test"
```

---

## Task 2: Implement `Inference::OverlapAddStreamer` (constant-memory overlap-add)

**Files:**
- Modify: `include/bs_roformer/inference.h`
- Modify: `src/inference.cpp`

### Step 1: Add public class declaration (header)

In `include/bs_roformer/inference.h`, add a new public helper class (additive API):

```cpp
class Inference {
public:
    using CancelCallback = std::function<bool()>;
    using ModelCallback = std::function<std::vector<std::vector<float>>(const std::vector<float>&)>;

    class OverlapAddStreamer {
    public:
        OverlapAddStreamer(int chunk_size, int num_overlap, int num_stems, ModelCallback model_func);

        // Feed interleaved stereo samples (size must be even). Returns ready output (may be empty).
        std::vector<std::vector<float>> Push(const std::vector<float>& input_audio);

        // Finish stream. After calling this, Push() must throw.
        std::vector<std::vector<float>> Finalize();

    private:
        int chunk_size_;
        int num_overlap_;
        int step_;
        int border_;
        int fade_size_;
        int num_stems_;
        ModelCallback model_func_;
        bool finalized_ = false;

        // Padding decision (matches ProcessOverlapAdd)
        bool decided_pad_ = false;
        bool do_pad_ = false;
        int64_t total_input_samples_ = 0; // per-channel sample count (not interleaved)

        // Pre-buffer to decide do_pad and build left reflection padding
        std::vector<float> prebuffer_; // interleaved

        // Main input buffer in padded domain (interleaved), from which we extract chunks
        std::vector<float> input_buffer_;

        // Sliding overlap-add window (padded domain), length chunk_size_ samples per channel
        std::vector<std::vector<float>> accum_; // [stems][chunk_size_*2]
        std::vector<float> counter_;            // [chunk_size_] (per-sample, not per-channel)

        // Window base (size chunk_size_)
        std::vector<float> window_base_;

        // Current offset in padded domain (per-channel samples)
        int64_t current_offset_ = 0;

        // How many padded samples have been “cropped away” on the left (pad_l)
        int64_t crop_left_ = 0;

        // Total padded samples processed so far (in padded domain)
        int64_t produced_padded_samples_ = 0;

        // Internal helpers
        void EnsurePaddingDecisionOrBuffer();
        void MaybeStartAfterPadDecision();
        std::vector<float> ExtractNextChunkOrEmpty(bool flushing);
        void AccumulateChunk(const std::vector<std::vector<float>>& chunk_out, bool is_first, bool is_last);
        std::vector<std::vector<float>> ExtractReadyOutput(int n_samples_ready);
    };

    // ... existing Inference API ...
};
```

### Step 2: Minimal implementation to make test compile (still failing)

In `src/inference.cpp`, add stub methods that compile but return empty output. Rebuild and run the new test to ensure it fails at runtime (diff mismatch).

### Step 3: Implement full streaming overlap-add (make test pass)

Implement behavior to match `Inference::ProcessOverlapAdd(...)`:

Key rules to replicate from `src/inference.cpp:758+`:
1. `channels = 2`, and input must be interleaved stereo (even length).
2. `step = chunk_size / num_overlap`, `border = chunk_size - step`, `fade_size = chunk_size / 10`.
3. `do_pad = (n_input_samples > 2 * border) && (border > 0)`.
4. If `do_pad`:
   - `pad_l = pad_r = border`.
   - left pad is reflect: `padded[pad_l-1-i] = input[1+i]` (clamp as in current code)
   - right pad is reflect: `padded[pad_l+n_input+i] = input[n_input-2-i]` (clamp)
5. Chunk extraction at offset `i` always yields `chunk_size` samples per channel:
   - If the remaining part is short (`part_len < chunk_size`), copy what exists and:
     - If `part_len > chunk_size/2 + 1`, reflect-pad right **within the chunk**; else leave zeros.
6. Windowing:
   - Start from `window_base_` (computed like existing `GetWindow(chunk_size, fade_size)`).
   - First chunk (`i==0`): force fade-in to 1 (`window[0:fade_size]=1`).
   - Last chunk (`i + step >= total_length`): force fade-out to 1 (`window[end-fade_size:]=1`).
7. Overlap-add and normalization match existing logic.
8. Crop output back to real (non-padded) length.

**Streaming trick:** Maintain a sliding window `accum_` / `counter_` of length `chunk_size`:
- After processing each chunk at offset `current_offset_`, you can safely output the first `step_` samples of `accum_` because future chunks start at `+step_` and cannot touch those samples.
- Then shift the window left by `step_` (memmove) and zero the newly freed tail.

**Implementation note (cropping):**
- When `do_pad_` becomes true, set `crop_left_ = border_` and suppress output until the first `crop_left_` padded samples have been dropped.
- Track `total_input_samples_` (per-channel) from pushed audio; on `Finalize()`, stop output at exactly `total_input_samples_`.

To keep work incremental (2–5 minutes per step), implement in the following micro-steps:

#### Step 3.1: Constructor + invariants

Implement `OverlapAddStreamer::OverlapAddStreamer(...)` to:
- Validate `chunk_size > 0`, `num_overlap >= 1`, `chunk_size % num_overlap == 0`.
- Compute `step_`, `border_`, `fade_size_`.
- Allocate:
  - `accum_ = vector(num_stems_, vector<float>(chunk_size_ * 2, 0))`
  - `counter_ = vector<float>(chunk_size_, 0)` (per-sample)
  - `window_base_ = GetWindow(chunk_size_, fade_size_)`

Run: `ctest ... -R test_streaming_overlap_add`  
Expected: still FAIL (logic not implemented), but compiles.

#### Step 3.2: Basic buffering + argument validation in `Push()`

Implement `Push()` to:
- Throw if `finalized_`.
- Validate `input_audio.size() % 2 == 0`.
- Update `total_input_samples_ += input_audio.size() / 2`.
- Append `input_audio` into `prebuffer_` if padding decision not made, else into `input_buffer_`.
- Return empty output for now.

Run the test again (still FAIL).

#### Step 3.3: Decide `do_pad` and materialize left padding (exactly like batch)

Implement a helper that, once `prebuffer_` has more than `2 * border_` samples-per-channel (or when flushing), decides:
- `do_pad_ = (total_input_samples_ > 2 * border_) && (border_ > 0)`
- If `do_pad_` is true:
  - Compute `crop_left_ = border_`.
  - Create `left_pad` (size `border_ * 2` floats interleaved) using the same reflection rules as `ProcessOverlapAdd`.
  - Set `input_buffer_ = left_pad + prebuffer_` (then clear `prebuffer_`).
- Else:
  - Set `crop_left_ = 0`.
  - Set `input_buffer_ = prebuffer_` (then clear `prebuffer_`).

Run: test should still FAIL, but now the first chunk extraction can be correct.

#### Step 3.4: Implement `ExtractNextChunkOrEmpty(flushing)`

Implement chunk extraction in the *padded domain*:
- Track `current_offset_` in **padded samples-per-channel**, starting at `0`.
- Compute `available_padded_samples = input_buffer_.size() / 2`.
- If not `flushing` and `available_padded_samples < current_offset_ + chunk_size_`, return `{}` (wait for more input).
- Else compute `part_len = min(chunk_size_, available_padded_samples - current_offset_)` (if negative => `{}`).
- Build `chunk_in` of size `chunk_size_ * 2` (float), zero-initialized.
- Copy `part_len` samples into `chunk_in` from `input_buffer_` starting at `current_offset_`.
- If `part_len < chunk_size_` and `part_len > chunk_size_/2 + 1`, reflect-pad right inside `chunk_in` (match batch code).

Run test (still FAIL).

#### Step 3.5: Implement `AccumulateChunk(...)` (window + accum + counter)

Implement accumulation into `accum_`/`counter_` (both in the padded domain):
- `window = window_base_` then adjust for:
  - `is_first`: force `window[0:fade_size_]=1`
  - `is_last`: force `window[end-fade_size_:]=1`
- For `k in [0, part_len)`:
  - `w = window[k]`
  - For each stem `s`: add `chunk_out[s][k*2+ch] * w` into `accum_[s][k*2+ch]`
  - `counter_[k] += w`

Important: `counter_` is per-sample; apply same `w` for both channels.

Run test (likely still FAIL, but output starts to appear).

#### Step 3.6: Implement “emit step + shift” + cropping

After each chunk is accumulated:
- Determine “ready padded samples”: `ready = step_`
- Convert padded-ready into real-ready by applying left crop:
  - If `crop_left_ > 0`, discard output until `produced_padded_samples_ >= crop_left_`
- Implement `ExtractReadyOutput(ready)`:
  - Normalize: `sample = accum / max(counter, 1e-4)` (match batch’s guard)
  - Return interleaved stereo vectors per stem for the ready portion only
- Then shift:
  - memmove `accum_` left by `step_` samples (i.e., `step_*2` floats)
  - memmove `counter_` left by `step_` samples
  - zero the tail region `[chunk_size_-step_, chunk_size_)`
  - `produced_padded_samples_ += step_`
  - `current_offset_ += step_`

Run test; expect closer, but edge cases may still fail.

#### Step 3.7: Implement `Finalize()` (right reflection padding + last window)

`Finalize()` must:
- Mark `finalized_ = true`.
- Ensure padding decision is made (if never decided, decide with current totals).
- If `do_pad_`:
  - Append right reflection padding (size `border_*2`) to `input_buffer_` using the same rules as batch.
- Now repeatedly:
  - Extract chunk with `flushing=true`.
  - Determine `is_last` using the batch condition: `current_offset_ + step_ >= total_length_padded`
  - Accumulate, emit step, shift until `current_offset_ >= total_length_padded`
- Stop output at exactly `total_input_samples_` samples-per-channel (crop right).

Run test; expected: PASS.

### Step 4: Run tests

Run:

```bash
cmake --build build --config Release --parallel
ctest --test-dir build -C Release -R test_streaming_overlap_add --output-on-failure
```

Expected: PASS.

### Step 5: Commit (optional)

```bash
git add include/bs_roformer/inference.h src/inference.cpp
git commit -m "feat: add constant-memory streaming overlap-add helper"
```

---

## Task 3: Add `Inference` streaming API (wraps `ProcessChunk()`)

**Files:**
- Modify: `include/bs_roformer/inference.h`
- Modify: `src/inference.cpp`

### Step 1: Add API signatures

In `include/bs_roformer/inference.h`, add:

```cpp
class Inference {
public:
    struct StreamContext {
        std::unique_ptr<OverlapAddStreamer> oa;
        bool finalized = false;
    };

    std::unique_ptr<StreamContext> CreateStream(
        int chunk_size = -1,
        int num_overlap = -1
    );

    // Feed interleaved stereo samples, get ready output (may be empty)
    std::vector<std::vector<float>> ProcessStream(
        StreamContext& ctx,
        const std::vector<float>& input_chunk
    );

    std::vector<std::vector<float>> FinalizeStream(StreamContext& ctx);
};
```

### Step 2: Implement using `OverlapAddStreamer` + `ProcessChunk()`

In `src/inference.cpp`:
- `CreateStream()`:
  - Resolve defaults from model (`GetDefaultChunkSize()`, `GetDefaultNumOverlap()`).
  - Create `OverlapAddStreamer(chunk_size, num_overlap, GetNumStems(), model_func)` where:

```cpp
auto model_func = [this](const std::vector<float>& chunk) {
    return this->ProcessChunk(chunk);
};
```

- `ProcessStream()`:
  - Throw if `ctx.finalized`.
  - Return `ctx.oa->Push(input_chunk)`.
- `FinalizeStream()`:
  - Throw if `ctx.finalized`.
  - Set `ctx.finalized=true` and return `ctx.oa->Finalize()`.

### Step 3: Add a small smoke test (no model dependency)

We can’t call `CreateStream()` without a model, but we can still add a compile-only test or defer. Keep Task 1/2 as the core regression test for overlap-add correctness.

### Step 4: Commit (optional)

```bash
git add include/bs_roformer/inference.h src/inference.cpp
git commit -m "feat: add streaming inference API (CreateStream/ProcessStream/FinalizeStream)"
```

---

## Task 4: Implement streaming WAV read/write path in CLI (no full-file buffers)

**Files:**
- Modify: `cli/main.cpp`
- Modify: `src/audio.cpp` (optional, if you want helper functions)
- Modify: `include/bs_roformer/audio.h` (optional, if you want helper classes)

### Step 1: Add CLI flag + choose defaults

In `cli/main.cpp`:
- Streaming is the default (no flag needed).
- Add option: `--no-stream` to fall back to the legacy full-load path (debug only; uses more RAM).

### Step 2: Implement streaming WAV reader (dr_wav)

In `cli/main.cpp` (CLI already includes `src` and `third_party` include dirs), add:

```cpp
#include "dr_libs/dr_wav.h"
```

Implement in small sub-steps (compile after each):

#### Step 2.1: Open WAV + validate format

- `drwav_init_file(...)`
- Validate:
  - `sampleRate == engine.GetSampleRate()` (44100)
  - `channels` is `1` or `2`
  - (optional) `totalPCMFrameCount > 0` for progress

Build:

```bash
cmake --build build --config Release --parallel
```

Expected: CLI builds.

#### Step 2.2: Create stream + open stem writers

- Create streaming context:
  - `auto stream = engine.CreateStream(chunk_size, num_overlap);`
- Determine stems via `engine.GetNumStems()`.
- Open `drwav` writers (one file per stem) **before** the read loop:
  - Use the current naming scheme in `cli/main.cpp:153+` (`_stem_i` insertion).
  - Use IEEE float, stereo, sampleRate=required_sr.

#### Step 2.3: Implement read → convert → ProcessStream → write loop

- Read frames in bounded blocks (recommend: 1 second = `44100` frames).
- Convert to interleaved stereo float buffer:
  - mono: duplicate on the fly
  - stereo: memcpy
- Call `engine.ProcessStream(*stream, stereo_buf)`
- For each stem returned, call `drwav_write_pcm_frames(...)`

#### Step 2.4: Finalize + close

- After EOF:
  - `auto tail = engine.FinalizeStream(*stream);`
  - write tail for each stem
- `drwav_uninit(&wav)` and `drwav_uninit(&writers[s])`

#### Step 2.5: Progress + cancellation (optional)

- Progress: use `frames_read_total / total_frames` and reuse existing progress bar callback.
- Cancellation: keep current behavior (no cancel), or add a keypress handler later (out of scope).

Pseudo-implementation (keep buffers small, e.g. 1s or `step` frames):

```cpp
drwav wav;
if (!drwav_init_file(&wav, input_path.c_str(), nullptr)) throw ...

const uint32_t in_ch = wav.channels;
const uint32_t sr = wav.sampleRate;
const drwav_uint64 total_frames = wav.totalPCMFrameCount;

if (sr != required_sr) throw ...
if (!(in_ch == 1 || in_ch == 2)) throw ...

auto stream = engine.CreateStream(chunk_size, num_overlap);
const int read_frames = 44100; // 1 second
std::vector<float> read_buf(read_frames * in_ch);
std::vector<float> stereo_buf(read_frames * 2);

// Open writers (one per stem) after we know stem count
const int stems = engine.GetNumStems();
std::vector<drwav> writers(stems);
// init writers: DR_WAVE_FORMAT_IEEE_FLOAT, channels=2, sampleRate=required_sr

drwav_uint64 frames_read_total = 0;
while (true) {
    drwav_uint64 got = drwav_read_pcm_frames_f32(&wav, read_frames, read_buf.data());
    if (got == 0) break;
    frames_read_total += got;

    // mono->stereo or pass-through
    stereo_buf.resize(got * 2);
    if (in_ch == 1) {
        for (uint64_t i=0;i<got;++i) {
            float x = read_buf[i];
            stereo_buf[i*2+0]=x;
            stereo_buf[i*2+1]=x;
        }
    } else {
        // in_ch==2, interleaved
        std::memcpy(stereo_buf.data(), read_buf.data(), got * 2 * sizeof(float));
    }

    auto out = engine.ProcessStream(*stream, stereo_buf);
    // out: [stems][samples] where samples are interleaved stereo floats
    // Convert float count -> frames: out[s].size()/2
    for (int s=0;s<stems;++s) {
        if (out.size() <= (size_t)s) continue;
        drwav_write_pcm_frames(&writers[s], out[s].size()/2, out[s].data());
    }

    if (progress_callback && total_frames > 0) {
        progress_callback((float)frames_read_total / (float)total_frames);
    }
}

auto tail = engine.FinalizeStream(*stream);
for (int s=0;s<stems;++s) {
    if (tail.size() <= (size_t)s) continue;
    drwav_write_pcm_frames(&writers[s], tail[s].size()/2, tail[s].data());
}

drwav_uninit(&wav);
for (auto& w: writers) drwav_uninit(&w);
```

### Step 3: Verify correctness (manual)

Run old path and new streaming path on a short audio and compare outputs (hash or waveform diff):
- Old: `bs_roformer-cli model.gguf in.wav out.wav --no-stream`
- New (default): `bs_roformer-cli model.gguf in.wav out.wav`

Expected: outputs match within float tolerance; no audible seam artifacts.

### Step 4: Commit (optional)

```bash
git add cli/main.cpp
git commit -m "feat(cli): add streaming wav read/write to reduce RAM"
```

---

## Task 5: Make GGML graph context configurable + safer (COMPLETED)

**Files:**
- Modify: `src/inference.cpp`

### Implementation notes

- Default graph ctx size is **32MB** (metadata only; tensor storage is allocated by `ggml_gallocr`).
- Config via env var: `BSR_GGML_GRAPH_CTX_MB` (min **16MB**, max **512MB**).
- Safe fallback: tries a small candidate set in order (primary → 32MB → 64MB → 128MB → 16MB).
- Robustness: uses an **external backing buffer** (`Inference::ctx_mem_`) passed to `ggml_init(...)` so ggml does not call its own allocator (avoids `GGML_ASSERT(ctx->mem_buffer != NULL)` on allocation failure).

### Verification

Run:

```bash
ctest --test-dir build -C Release -R test_inference --output-on-failure
```

Expected: no regressions.

### Step 4: Commit (optional)

```bash
git add src/inference.cpp
git commit -m "perf(mem): reduce ggml graph ctx size and make configurable"
```

---

## Task 6: Full verification checklist (manual + automated)

**Automated**
- Build (PowerShell + VS Build Tools):
  - `.\build.ps1`
- Build + tests (enable tests explicitly):
  - `cmd /c "\"C:\\Program Files (x86)\\Microsoft Visual Studio\\18\\BuildTools\\VC\\Auxiliary\\Build\\vcvarsall.bat\" x64 && cmake -S . -B build -G \"Ninja Multi-Config\" -DBSR_BUILD_TESTS=ON -DBSR_BUILD_CLI=ON"`
  - `cmd /c "\"C:\\Program Files (x86)\\Microsoft Visual Studio\\18\\BuildTools\\VC\\Auxiliary\\Build\\vcvarsall.bat\" x64 && cmake --build build --config Release --parallel"`
  - `ctest --test-dir build -C Release --output-on-failure`

**Manual (Windows)**
1. 25-min WAV run:
   - `bs_roformer-cli model.gguf 25min.wav out.wav`
2. Observe peak RAM in Task Manager (or `Get-Process bs_roformer-cli | Select-Object -Expand WorkingSet64` in a loop).
3. Spot-check output seams at chunk boundaries (listen to start/end + random boundaries).

---

## Task 7: Pipelined streaming inference (reduce GPU bubble)

**Goal:** Keep the constant-memory streaming path, but reduce GPU idle time by overlapping:
- CPU pre-process (`STFT` + input packing)
- GPU compute (`ggml_backend_graph_compute`)
- CPU post-process (`ISTFT` + overlap-add accumulate)

**Root cause (current code):**
- Streaming currently uses `OverlapAddStreamer` + `Inference::ProcessChunk()` which is fully serial:
  - `PreProcessChunk` → `RunInference` → `PostProcessChunk`
- So GPU waits during STFT/ISTFT (“GPU bubble”).

**Approach (recommended):**
- Add a **3-stage pipeline** inside the streaming session (bounded in-flight chunks, default depth=3).
- Preserve ordering (FIFO) so overlap-add accumulation stays deterministic.
- Default enable for `CreateStream()` / CLI streaming path; keep a debug fallback to serial streaming.

**Files:**
- Modify: `include/bs_roformer/inference.h`
- Modify: `src/inference.cpp`
- Modify (optional): `cli/main.cpp`
- Test: `tests/test_streaming_overlap_add.cpp` (extend)

### Step 1: Add a test that exercises “schedule/consume” overlap-add (no model dependency)

Extend `tests/test_streaming_overlap_add.cpp`:
- Add a second codepath that uses the new manual scheduling API:
  1) `Feed(...)`
  2) `TryScheduleNext(...)` into an in-flight queue (depth=3)
  3) “Process” via identity callback
  4) `ConsumeScheduled(...)` in-order
  5) Compare against `Inference::ProcessOverlapAdd(...)`

Expected: exact match within `1e-6`.

### Step 2: Refactor `OverlapAddStreamer` to support pipelined scheduling

Add an internal/manual API to decouple:
- **Input feed / pad decision**
- **Chunk scheduling** (extract `chunk_in` + metadata)
- **Chunk consumption** (accumulate + emit ready output)

Keep existing `Push()` / `Finalize()` behavior unchanged by re-implementing them on top of the new primitives.

### Step 3: Implement `Inference::CreateStream()` pipelined session (default ON)

Implement a streaming session that:
- Uses the manual overlap-add scheduler to generate `chunk_in`.
- Runs 3 worker stages with bounded queues:
  - thread A: `PreProcessChunk`
  - thread B: `RunInference` (single thread; ggml backend not shared)
  - thread C: `PostProcessChunk`
- Returns output via overlap-add `ConsumeScheduled(...)` as chunks finish.

Suggested toggles:
- `Inference::CreateStream(..., bool pipelined=true)` (debug)
- Env: `BSR_STREAM_PIPELINE_DEPTH` (default `3`, clamp `[1, 8]`)

### Step 4: Optional CLI debug switch

Add `--no-pipeline` to force serial streaming (still constant-memory) for A/B comparisons.

### Step 5: Verification

- Unit tests:
  - `ctest --test-dir build -C Release -R test_streaming_overlap_add --output-on-failure`
- Functional:
  - Run a medium WAV and verify the output matches (hash / waveform diff) between:
    - serial streaming (`--no-pipeline`)
    - pipelined streaming (default)
- Perf:
  - Add optional timing logs (guarded by env var) around:
    - `PreProcessChunk`, `RunInference`, `PostProcessChunk`
  - Confirm GPU utilization improves (less bubble) in your profiler.

---

## Task 8: STFT/ISTFT hot-path optimization (reduce CPU time / GPU bubble)

**Goal:** Even with a pipelined stream, GPU can still idle if CPU pre/post is slow. Reduce CPU overhead by:
- Reusing large temporary buffers to avoid repeated allocations per chunk.
- Avoiding per-frame FFT “GetInstance” locking/lookup overhead.
- Dropping redundant intermediate buffers in postprocess.

**Files:**
- Modify: `src/stft.h`
- Modify: `src/inference.cpp`

**Status:** Implemented (2026-03-04).

**Verification:**
- `ctest --test-dir build -C Release --output-on-failure`

## Execution Handoff

Plan saved to `docs/plans/2026-03-04-streaming-inference-memory-b-plan.md`.

Two execution options:
1. **Parallel Session:** open a new session and use **superpowers:executing-plans** to run task-by-task.
2. **This session:** you tell me “go implement Task N”, and I’ll implement with check-ins after each task.
