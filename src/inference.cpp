#include "bs_roformer/inference.h"
#include "model.h"
#include "utils.h"
#include "stft.h"
#include <iostream>
#include <complex>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <chrono>
#include <future>
#include <iomanip>
#include <queue>
#include <sstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <exception>
#include <cstdlib>
#include <limits>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#endif

#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "ggml-cuda.h"
#include <cuda_runtime.h>
#endif

using Complex = std::complex<float>;
static constexpr const char* kInferenceCancelledMessage = "Inference cancelled";

// Helper forward decl
std::vector<float> GetWindow(int size, int fade_size);

static int GetStreamTimingLevel();
static void LogStreamTimingLine(const std::string& line);

static std::mutex g_stream_timing_mutex;

static bool WantCudaHostRegister() {
#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const char* env = std::getenv("GGML_CUDA_REGISTER_HOST");
    if (!env || !*env) return false;
    // Treat "0" as disabled.
    return std::strcmp(env, "0") != 0;
#else
    return false;
#endif
}

static bool WantCudaPinnedStaging() {
#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (std::getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return false;
    }

    const char* env = std::getenv("BSR_CUDA_PINNED_STAGING");
    if (!env || !*env) {
        // Default: enabled for CUDA to avoid pageable async staging growth.
        return true;
    }

    char* end = nullptr;
    long v = std::strtol(env, &end, 10);
    if (end == env) {
        // Non-numeric but set => enabled
        return true;
    }
    return v > 0;
#else
    return false;
#endif
}

static void* TryCudaMallocHost(size_t bytes) {
#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, bytes);
    if (err != cudaSuccess) {
        // clear the error
        (void) cudaGetLastError();
        return nullptr;
    }
    return ptr;
#else
    (void) bytes;
    return nullptr;
#endif
}

static void TryCudaFreeHost(void* ptr) {
#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (ptr == nullptr) return;
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaSuccess) {
        // clear the error
        (void) cudaGetLastError();
    }
#else
    (void) ptr;
#endif
}

static bool TryCudaHostRegister(void* buffer, size_t bytes, unsigned flags) {
#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (buffer == nullptr || bytes == 0) return false;
    cudaError_t err = cudaHostRegister(buffer, bytes, flags);
    if (err != cudaSuccess) {
        // clear the error
        (void) cudaGetLastError();
        return false;
    }
    return true;
#else
    (void) buffer;
    (void) bytes;
    (void) flags;
    return false;
#endif
}

static void TryCudaHostUnregister(void* buffer) {
#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (buffer == nullptr) return;
    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void) cudaGetLastError();
    }
#else
    (void) buffer;
#endif
}

static void EnsureCudaHostRegistered(void* buffer,
                                     size_t bytes,
                                     unsigned flags,
                                     void*& reg_ptr,
                                     size_t& reg_bytes) {
    if (!WantCudaHostRegister()) return;
    if (buffer == nullptr || bytes == 0) return;

    if (reg_ptr == buffer && reg_bytes == bytes) {
        return;
    }

    if (reg_ptr != nullptr) {
        TryCudaHostUnregister(reg_ptr);
        reg_ptr = nullptr;
        reg_bytes = 0;
    }

    if (TryCudaHostRegister(buffer, bytes, flags)) {
        reg_ptr = buffer;
        reg_bytes = bytes;
    }
}

static int GetStreamMemLevel() {
    const char* env = std::getenv("BSR_STREAM_MEM");
    if (!env || !*env) return 0;

    char* end = nullptr;
    long v = std::strtol(env, &end, 10);
    if (end == env) {
        // Non-numeric but set => enabled
        return 1;
    }
    if (v <= 0) return 0;
    if (v > 3) v = 3;
    return static_cast<int>(v);
}

struct ProcessMemInfo {
    size_t working_set_bytes = 0;
    size_t private_bytes = 0;
};

static ProcessMemInfo GetProcessMemInfo() {
    ProcessMemInfo info{};
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
        info.working_set_bytes = static_cast<size_t>(pmc.WorkingSetSize);
        info.private_bytes = static_cast<size_t>(pmc.PrivateUsage);
    }
#endif
    return info;
}

Inference::ChunkState::~ChunkState() {
    if (cuda_reg_stft_ptr != nullptr) {
        TryCudaHostUnregister(cuda_reg_stft_ptr);
        cuda_reg_stft_ptr = nullptr;
        cuda_reg_stft_bytes = 0;
    }
    if (cuda_reg_mask_ptr != nullptr) {
        TryCudaHostUnregister(cuda_reg_mask_ptr);
        cuda_reg_mask_ptr = nullptr;
        cuda_reg_mask_bytes = 0;
    }
}

static int GetMaskStatsLevel() {
    const char* env = std::getenv("BSR_MASK_STATS");
    if (!env || !*env) return 0;

    char* end = nullptr;
    long v = std::strtol(env, &end, 10);
    if (end == env) return 1;
    if (v <= 0) return 0;
    if (v > 2) v = 2;
    return static_cast<int>(v);
}

struct FloatStats {
    double mean_abs = 0.0;
    float max_abs = 0.0f;
    float min = 0.0f;
    float max = 0.0f;
    size_t count = 0;
};

static FloatStats ComputeFloatStats(const std::vector<float>& v) {
    FloatStats s;
    s.count = v.size();
    if (v.empty()) return s;

    float min_v = v[0];
    float max_v = v[0];
    float max_abs = std::fabs(v[0]);
    long double sum_abs = static_cast<long double>(max_abs);

    for (size_t i = 1; i < v.size(); ++i) {
        float x = v[i];
        min_v = std::min(min_v, x);
        max_v = std::max(max_v, x);
        float a = std::fabs(x);
        max_abs = std::max(max_abs, a);
        sum_abs += static_cast<long double>(a);
    }

    s.min = min_v;
    s.max = max_v;
    s.max_abs = max_abs;
    s.mean_abs = static_cast<double>(sum_abs / static_cast<long double>(v.size()));
    return s;
}

std::vector<float> GetWindow(int size, int fade_size) {
    std::vector<float> window(size, 1.0f);
    // Match Python: torch.linspace(0, 1, fade_size) and torch.linspace(1, 0, fade_size)
    // linspace includes both endpoints, so we divide by (fade_size - 1)
    for (int i = 0; i < fade_size; ++i) {
        // fadein[i] = i / (fade_size - 1), ranges from 0.0 to 1.0 inclusive
        // fadeout[i] = 1 - i / (fade_size - 1), ranges from 1.0 to 0.0 inclusive
        float fadein = (fade_size > 1) ? (float)i / (fade_size - 1) : 1.0f;
        float fadeout = (fade_size > 1) ? 1.0f - (float)i / (fade_size - 1) : 1.0f;
        window[i] *= fadein;                     // Start of window: fade in
        window[size - fade_size + i] *= fadeout; // End of window: fade out
    }
    return window;
}

static int GetStreamTimingLevel() {
    const char* env = std::getenv("BSR_STREAM_TIMING");
    if (!env || !*env) return 0;

    char* end = nullptr;
    long v = std::strtol(env, &end, 10);
    if (end == env) {
        // Non-numeric but set => enabled
        return 1;
    }
    if (v <= 0) return 0;
    if (v > 3) v = 3;
    return static_cast<int>(v);
}

static void LogStreamTimingLine(const std::string& line) {
    std::lock_guard<std::mutex> lock(g_stream_timing_mutex);
    std::cerr << line << std::endl;
}


Inference::Inference(const std::string& model_path) {
    model_ = std::make_unique<BSRoformer>();
    model_->Initialize(model_path);

    // Precompute Hann window once (used in STFT/ISTFT for every chunk)
    const int win_length = model_->GetWinLength();
    hann_window_.resize(win_length);
    stft::hann_window(hann_window_.data(), win_length);
}

int Inference::GetDefaultChunkSize() const {
    return model_->GetDefaultChunkSize();
}

int Inference::GetDefaultNumOverlap() const {
    return model_->GetDefaultNumOverlap();
}

int Inference::GetSampleRate() const {
    return model_->GetSampleRate();
}

int Inference::GetNumStems() const {
    return model_->GetNumStems();
}

Inference::~Inference() {
    if (cuda_pinned_in_ != nullptr) {
        TryCudaFreeHost(cuda_pinned_in_);
        cuda_pinned_in_ = nullptr;
        cuda_pinned_in_bytes_ = 0;
    }
    if (cuda_pinned_out_ != nullptr) {
        TryCudaFreeHost(cuda_pinned_out_);
        cuda_pinned_out_ = nullptr;
        cuda_pinned_out_bytes_ = 0;
    }
    if (cuda_pinned_pos_time_ != nullptr) {
        TryCudaFreeHost(cuda_pinned_pos_time_);
        cuda_pinned_pos_time_ = nullptr;
        cuda_pinned_pos_time_bytes_ = 0;
    }
    if (cuda_pinned_pos_freq_ != nullptr) {
        TryCudaFreeHost(cuda_pinned_pos_freq_);
        cuda_pinned_pos_freq_ = nullptr;
        cuda_pinned_pos_freq_bytes_ = 0;
    }

    if (allocr_) ggml_gallocr_free(allocr_);
    if (ctx_) ggml_free(ctx_);
    // gf_ is part of ctx_, tensor pointers are part of ctx_
}

bool Inference::EnsureGraph(int n_frames) {
    if (n_frames == cached_n_frames_ && ctx_ != nullptr) {
        return true;
    }
    
    std::cout << "[Inference] Building graph for n_frames=" << n_frames << std::endl;

    auto reset_graph_state = [&]() {
        if (allocr_) { ggml_gallocr_free(allocr_); allocr_ = nullptr; }
        if (ctx_) { ggml_free(ctx_); ctx_ = nullptr; }

        gf_ = nullptr;
        input_tensor_ = nullptr;
        pos_time_ = nullptr;
        pos_freq_ = nullptr;
        mask_out_tensor_ = nullptr;

        cached_n_frames_ = -1;
        uploaded_pos_n_frames_ = -1;
        ctx_mem_.clear();

        if (cuda_pinned_in_ != nullptr) {
            TryCudaFreeHost(cuda_pinned_in_);
            cuda_pinned_in_ = nullptr;
            cuda_pinned_in_bytes_ = 0;
        }
        if (cuda_pinned_out_ != nullptr) {
            TryCudaFreeHost(cuda_pinned_out_);
            cuda_pinned_out_ = nullptr;
            cuda_pinned_out_bytes_ = 0;
        }
        if (cuda_pinned_pos_time_ != nullptr) {
            TryCudaFreeHost(cuda_pinned_pos_time_);
            cuda_pinned_pos_time_ = nullptr;
            cuda_pinned_pos_time_bytes_ = 0;
        }
        if (cuda_pinned_pos_freq_ != nullptr) {
            TryCudaFreeHost(cuda_pinned_pos_freq_);
            cuda_pinned_pos_freq_ = nullptr;
            cuda_pinned_pos_freq_bytes_ = 0;
        }
    };

    reset_graph_state();

    const size_t MB = 1024ull * 1024ull;
    const size_t kDefaultCtxMb = 32;
    const size_t kMinCtxMb = 16;
    const size_t kMaxCtxMb = 512;

    size_t primary_ctx_bytes = kDefaultCtxMb * MB;
    if (const char* env = std::getenv("BSR_GGML_GRAPH_CTX_MB")) {
        char* end = nullptr;
        const unsigned long long mb = std::strtoull(env, &end, 10);
        if (end != env && mb > 0) {
            if (mb < kMinCtxMb) {
                std::cerr << "[Inference] BSR_GGML_GRAPH_CTX_MB=" << mb << " is too small; using minimum "
                          << kMinCtxMb << "MB" << std::endl;
                primary_ctx_bytes = kMinCtxMb * MB;
            } else if (mb > kMaxCtxMb) {
                std::cerr << "[Inference] BSR_GGML_GRAPH_CTX_MB=" << mb << " is too large; clamping to "
                          << kMaxCtxMb << "MB" << std::endl;
                primary_ctx_bytes = kMaxCtxMb * MB;
            } else if (mb > (std::numeric_limits<size_t>::max() / MB)) {
                std::cerr << "[Inference] BSR_GGML_GRAPH_CTX_MB overflows size_t; using default "
                          << kDefaultCtxMb << "MB" << std::endl;
                primary_ctx_bytes = kDefaultCtxMb * MB;
            } else {
                primary_ctx_bytes = static_cast<size_t>(mb) * MB;
            }
        }
    }

    auto add_unique = [](std::vector<size_t>& dst, size_t v) {
        if (v == 0) return;
        if (std::find(dst.begin(), dst.end(), v) != dst.end()) return;
        dst.push_back(v);
    };

    std::vector<size_t> candidates;
    add_unique(candidates, primary_ctx_bytes);
    add_unique(candidates, kDefaultCtxMb * MB);
    add_unique(candidates, 64ull * MB);
    add_unique(candidates, 128ull * MB);
    add_unique(candidates, kMinCtxMb * MB);

    for (size_t candidate_bytes : candidates) {
        const size_t mem_size = GGML_PAD(candidate_bytes, GGML_MEM_ALIGN);
        const size_t mem_mb = mem_size / MB;

        std::cout << "[Inference] Trying ggml graph ctx=" << mem_mb << "MB" << std::endl;

        try {
            ctx_mem_.assign(mem_size, 0);
        } catch (const std::bad_alloc&) {
            std::cerr << "[Inference] Failed to allocate host ctx buffer (" << mem_mb << "MB), trying next..." << std::endl;
            ctx_mem_.clear();
            continue;
        }

        struct ggml_init_params ctx_params = { mem_size, ctx_mem_.data(), true };
        ctx_ = ggml_init(ctx_params);

        gf_ = ggml_new_graph_custom(ctx_, 65536, false);

        int batch = 1;
        int total_dim_input = model_->GetTotalDimInput();

        input_tensor_ = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, total_dim_input, n_frames, batch);
        pos_time_ = nullptr;
        pos_freq_ = nullptr;
        mask_out_tensor_ = nullptr;

        if (!input_tensor_) {
            std::cerr << "[Inference] Failed to allocate input tensor (ctx too small?)" << std::endl;
            reset_graph_state();
            continue;
        }
        ggml_set_input(input_tensor_);

        // BandSplit -> Transformers -> MaskEstimator
        ggml_tensor* band_out = model_->BuildBandSplitGraph(ctx_, input_tensor_, gf_, n_frames, batch);
        if (!band_out) {
            std::cerr << "[Inference] Failed to build BandSplit graph" << std::endl;
            reset_graph_state();
            continue;
        }

        int n_bands = model_->GetNumBands();
        pos_time_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_frames * n_bands);
        pos_freq_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_bands * n_frames);
        if (!pos_time_ || !pos_freq_) {
            std::cerr << "[Inference] Failed to allocate position tensors (ctx too small?)" << std::endl;
            reset_graph_state();
            continue;
        }
        ggml_set_input(pos_time_);
        ggml_set_input(pos_freq_);

        ggml_tensor* trans_out = model_->BuildTransformersGraph(ctx_, band_out, gf_, pos_time_, pos_freq_, n_frames, batch);
        if (!trans_out) {
            std::cerr << "[Inference] Failed to build Transformers graph" << std::endl;
            reset_graph_state();
            continue;
        }

        mask_out_tensor_ = model_->BuildMaskEstimatorGraph(ctx_, trans_out, gf_, n_frames, batch);
        if (!mask_out_tensor_) {
            std::cerr << "[Inference] Failed to build MaskEstimator graph" << std::endl;
            reset_graph_state();
            continue;
        }

        // Allocate compute buffer (VRAM)
        allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model_->GetBackend()));
        if (!allocr_ || !ggml_gallocr_alloc_graph(allocr_, gf_)) {
            std::cerr << "[Inference] Failed to allocate graph VRAM" << std::endl;
            reset_graph_state();
            continue;
        }

#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
            ggml_backend_t backend = model_ ? model_->GetBackend() : nullptr;
            if (backend && ggml_backend_is_cuda(backend) && WantCudaPinnedStaging()) {
                const size_t in_bytes = ggml_nbytes(input_tensor_);
                if (cuda_pinned_in_bytes_ != in_bytes) {
                    if (cuda_pinned_in_ != nullptr) TryCudaFreeHost(cuda_pinned_in_);
                    cuda_pinned_in_ = TryCudaMallocHost(in_bytes);
                    cuda_pinned_in_bytes_ = (cuda_pinned_in_ != nullptr) ? in_bytes : 0;
                }

                const size_t out_bytes = ggml_nbytes(mask_out_tensor_);
                if (cuda_pinned_out_bytes_ != out_bytes) {
                    if (cuda_pinned_out_ != nullptr) TryCudaFreeHost(cuda_pinned_out_);
                    cuda_pinned_out_ = TryCudaMallocHost(out_bytes);
                    cuda_pinned_out_bytes_ = (cuda_pinned_out_ != nullptr) ? out_bytes : 0;
                }

                // Pinned staging for RoPE position tensors. These are small (~hundreds of KB), but if they come
                // from pageable memory, cudaMemcpyAsync may allocate internal pinned staging buffers that can
                // appear as gradual host memory growth on some drivers for long runs.
                const size_t pos_time_bytes = ggml_nbytes(pos_time_);
                if (cuda_pinned_pos_time_bytes_ != pos_time_bytes) {
                    if (cuda_pinned_pos_time_ != nullptr) TryCudaFreeHost(cuda_pinned_pos_time_);
                    cuda_pinned_pos_time_ = TryCudaMallocHost(pos_time_bytes);
                    cuda_pinned_pos_time_bytes_ = (cuda_pinned_pos_time_ != nullptr) ? pos_time_bytes : 0;
                }
                const size_t pos_freq_bytes = ggml_nbytes(pos_freq_);
                if (cuda_pinned_pos_freq_bytes_ != pos_freq_bytes) {
                    if (cuda_pinned_pos_freq_ != nullptr) TryCudaFreeHost(cuda_pinned_pos_freq_);
                    cuda_pinned_pos_freq_ = TryCudaMallocHost(pos_freq_bytes);
                    cuda_pinned_pos_freq_bytes_ = (cuda_pinned_pos_freq_ != nullptr) ? pos_freq_bytes : 0;
                }
            }
        }
#endif

        cached_n_frames_ = n_frames;
        return true;
    }

    std::cerr << "[Inference] Failed to build graph for n_frames=" << n_frames
              << ". You can try setting BSR_GGML_GRAPH_CTX_MB (e.g. 32 or 64)." << std::endl;
    return false;
}

void Inference::ComputeSTFT(const std::vector<float>& input_audio,
                            std::vector<std::vector<float>>& stft_outputs,
                            int& n_frames) {
    int n_fft = model_->GetNFFT();
    int hop_length = model_->GetHopLength();
    int win_length = model_->GetWinLength();
    int n_freq = n_fft / 2 + 1;
    int channels = 2; 

    stft_outputs.resize(channels);
    int n_samples = input_audio.size() / channels;

    struct StftScratch {
        std::vector<float> ch0;
        std::vector<float> ch1;
    };
    static thread_local StftScratch tls;

    tls.ch0.resize(n_samples);
    tls.ch1.resize(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        const size_t base = static_cast<size_t>(i) * channels;
        tls.ch0[i] = input_audio[base + 0];
        tls.ch1[i] = input_audio[base + 1];
    }

    const float* channel_ptrs[2] = { tls.ch0.data(), tls.ch1.data() };

    for (int ch = 0; ch < channels; ++ch) {
        stft_outputs[ch].resize(n_freq * (n_samples / hop_length + 5) * 2);
        stft::compute_stft(channel_ptrs[ch], n_samples, n_fft, hop_length, win_length,
                           hann_window_.data(), true, stft_outputs[ch].data(), &n_frames);
    }
}

void Inference::PrepareModelInput(const std::vector<std::vector<float>>& stft_outputs,
                                  int n_frames,
                                  std::vector<float>& model_input_rearranged) {
    const std::vector<int>& freq_indices = model_->GetFreqIndices();
    int num_freq_indices = freq_indices.size();
    int total_dim_input = model_->GetTotalDimInput();
    int channels = 2;

    model_input_rearranged.resize(n_frames * total_dim_input);

    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int t = 0; t < n_frames; ++t) {
        for (int f = 0; f < num_freq_indices; ++f) {
            int idx = freq_indices[f];
            int raw_freq_idx = idx / channels;
            int ch = idx % channels;

            int in_idx_ch = (raw_freq_idx * n_frames + t) * 2;
            int out_idx = t * total_dim_input + f * 2;

            model_input_rearranged[out_idx + 0] = stft_outputs[ch][in_idx_ch + 0];
            model_input_rearranged[out_idx + 1] = stft_outputs[ch][in_idx_ch + 1];
        }
    }
}

void Inference::PostProcessAndISTFT(const std::vector<float>& mask_output,
                                    const std::vector<std::vector<float>>& stft_outputs,
                                    int n_frames,
                                    std::vector<std::vector<float>>& output_audio) {
    int n_fft = model_->GetNFFT();
    int hop_length = model_->GetHopLength();
    int win_length = model_->GetWinLength();
    int n_freq = n_fft / 2 + 1;
    int channels = 2;

    const std::vector<int>& freq_indices = model_->GetFreqIndices();
    int num_freq_indices = freq_indices.size();
    int mask_features = num_freq_indices * 2;
    int num_stems = model_->GetNumStems();
    // Tensor layout: [mask_features, num_stems, n_frames, batch]
    // GGML stride for time t is: mask_features * num_stems
    int stride_time = mask_features * num_stems;
    
    output_audio.resize(num_stems);

    int total_freq_stereo = n_freq * channels;

    struct PostProcessScratch {
        std::vector<Complex> masks_summed;
        std::vector<float> istft_in;
        std::vector<float> out_ch0;
        std::vector<float> out_ch1;
    };
    static thread_local PostProcessScratch tls;

    tls.masks_summed.resize(static_cast<size_t>(total_freq_stereo) * static_cast<size_t>(n_frames));
    tls.istft_in.resize(static_cast<size_t>(n_freq) * static_cast<size_t>(n_frames) * 2);

    auto& masks_summed = tls.masks_summed;
    auto& istft_in = tls.istft_in;
    
    // Process each stem
    for (int stem = 0; stem < num_stems; ++stem) {
        std::fill(masks_summed.begin(), masks_summed.end(), Complex(0.0f, 0.0f));
        
        #ifdef USE_OPENMP
        #pragma omp parallel for
        #endif
        for (int t = 0; t < n_frames; ++t) {
            // Base index for this frame and current stem
            int frame_offset = t * stride_time;
            int stem_offset = stem * mask_features;
            int base_offset = frame_offset + stem_offset;
            
            for (int f = 0; f < num_freq_indices; ++f) {
                int idx = base_offset + f * 2;
                int dst_idx_base = freq_indices[f];
                masks_summed[static_cast<size_t>(dst_idx_base) * static_cast<size_t>(n_frames) + static_cast<size_t>(t)] +=
                    Complex(mask_output[idx + 0], mask_output[idx + 1]);
            }
        }

        const std::vector<int>& num_bands_per_freq = model_->GetNumBandsPerFreq();
        #ifdef USE_OPENMP
        #pragma omp parallel for
        #endif
        for (int f = 0; f < n_freq; ++f) {
            float denom = (float)num_bands_per_freq[f];
            if (denom < 1e-8f) denom = 1e-8f;
            for (int ch = 0; ch < channels; ++ch) {
                int freq_stereo_idx = f * channels + ch;
                for (int t = 0; t < n_frames; ++t) {
                    masks_summed[static_cast<size_t>(freq_stereo_idx) * static_cast<size_t>(n_frames) + static_cast<size_t>(t)] /= denom;
                }
            }
        }

        int n_samples_out = 0;

        for (int ch = 0; ch < channels; ++ch) {
            std::vector<float>& output_channel = (ch == 0) ? tls.out_ch0 : tls.out_ch1;
            for (int f = 0; f < n_freq; ++f) {
                int freq_stereo_idx = f * channels + ch;
                for (int t = 0; t < n_frames; ++t) {
                    int dst_idx = (f * n_frames + t) * 2;

                    size_t mask_idx = static_cast<size_t>(freq_stereo_idx) * static_cast<size_t>(n_frames) + static_cast<size_t>(t);
                    size_t stft_idx = (static_cast<size_t>(f) * static_cast<size_t>(n_frames) + static_cast<size_t>(t)) * 2;
                    Complex stft_val(stft_outputs[ch][stft_idx + 0], stft_outputs[ch][stft_idx + 1]);
                    Complex masked = stft_val * masks_summed[mask_idx];
                    istft_in[dst_idx + 0] = masked.real();
                    istft_in[dst_idx + 1] = masked.imag();
                }
            }
            
            // Zero DC if enabled
            if (model_->GetZeroDC()) {
                for (int t = 0; t < n_frames; ++t) {
                    // f=0 is DC component
                    int dst_idx = (0 * n_frames + t) * 2; 
                    istft_in[dst_idx + 0] = 0.0f;
                    istft_in[dst_idx + 1] = 0.0f;
                }
            }
            
            int approx_len = (n_frames - 1) * hop_length + n_fft;
            output_channel.resize(approx_len + n_fft);
            stft::compute_istft(istft_in.data(), n_freq, n_frames, n_fft, hop_length, win_length, 
                                hann_window_.data(), true, approx_len, output_channel.data());
            if (ch == 0) n_samples_out = approx_len;
            output_channel.resize(n_samples_out);
        }

        output_audio[stem].resize(channels * n_samples_out);
        for (int i = 0; i < n_samples_out; ++i) {
            output_audio[stem][0 + i * channels] = tls.out_ch0[i];
            output_audio[stem][1 + i * channels] = tls.out_ch1[i];
        }
    }
}

std::vector<std::vector<float>> Inference::Process(const std::vector<float>& input_audio,
                                                   int chunk_size,
                                                   int num_overlap,
                                                   std::function<void(float)> progress_callback,
                                                   CancelCallback cancel_callback) {
    if (input_audio.empty()) return {};
    return ProcessOverlapAddPipelined(input_audio, chunk_size, num_overlap, progress_callback, cancel_callback);
}

// =================================================================================================
// Pipeline Stages
// =================================================================================================

void Inference::PreProcessChunkInto(ChunkState& state, std::vector<float> chunk_audio, int64_t id) {
    state.id = id;
    state.input_audio = std::move(chunk_audio);
    state.mask_output.clear();
    state.n_frames = 0;

    // Preserve per-stem output buffers to avoid per-chunk alloc/free churn (can make Windows Private Bytes climb).
    const int num_stems = model_ ? model_->GetNumStems() : 0;
    if (num_stems > 0) {
        if (state.final_audio.size() != static_cast<size_t>(num_stems)) {
            state.final_audio.resize(static_cast<size_t>(num_stems));
        }
        for (auto& stem_audio : state.final_audio) {
            stem_audio.clear();
        }
    } else {
        state.final_audio.clear();
    }

    if (state.input_audio.empty()) {
        state.stft_flattened.clear();
        return;
    }

    // 1. STFT
    ComputeSTFT(state.input_audio, state.stft_outputs, state.n_frames);

    const size_t required_stft = static_cast<size_t>(state.n_frames) * static_cast<size_t>(model_->GetTotalDimInput());

#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    ggml_backend_t backend = model_ ? model_->GetBackend() : nullptr;
    if (backend && ggml_backend_is_cuda(backend)) {
        // If a resize might reallocate, unregister first (must not free registered memory).
        if (state.cuda_reg_stft_ptr != nullptr && state.stft_flattened.capacity() < required_stft) {
            TryCudaHostUnregister(state.cuda_reg_stft_ptr);
            state.cuda_reg_stft_ptr = nullptr;
            state.cuda_reg_stft_bytes = 0;
        }
    }
#endif

    // 2. Prepare Input
    PrepareModelInput(state.stft_outputs, state.n_frames, state.stft_flattened);

#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (backend && ggml_backend_is_cuda(backend) && !state.stft_flattened.empty()) {
        EnsureCudaHostRegistered(state.stft_flattened.data(),
                                 state.stft_flattened.size() * sizeof(float),
                                 cudaHostRegisterPortable | cudaHostRegisterReadOnly,
                                 state.cuda_reg_stft_ptr,
                                 state.cuda_reg_stft_bytes);
    }
#endif
}

void Inference::PreProcessChunkInPlace(ChunkState& state) {
    state.mask_output.clear();
    state.n_frames = 0;

    // Preserve per-stem output buffers to avoid per-chunk alloc/free churn (can make Windows Private Bytes climb).
    const int num_stems = model_ ? model_->GetNumStems() : 0;
    if (num_stems > 0) {
        if (state.final_audio.size() != static_cast<size_t>(num_stems)) {
            state.final_audio.resize(static_cast<size_t>(num_stems));
        }
        for (auto& stem_audio : state.final_audio) {
            stem_audio.clear();
        }
    } else {
        state.final_audio.clear();
    }

    if (state.input_audio.empty()) {
        state.stft_flattened.clear();
        return;
    }

    // 1. STFT
    ComputeSTFT(state.input_audio, state.stft_outputs, state.n_frames);

    const size_t required_stft = static_cast<size_t>(state.n_frames) * static_cast<size_t>(model_->GetTotalDimInput());

#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    ggml_backend_t backend = model_ ? model_->GetBackend() : nullptr;
    if (backend && ggml_backend_is_cuda(backend)) {
        // If a resize might reallocate, unregister first (must not free registered memory).
        if (state.cuda_reg_stft_ptr != nullptr && state.stft_flattened.capacity() < required_stft) {
            TryCudaHostUnregister(state.cuda_reg_stft_ptr);
            state.cuda_reg_stft_ptr = nullptr;
            state.cuda_reg_stft_bytes = 0;
        }
    }
#endif

    // 2. Prepare Input
    PrepareModelInput(state.stft_outputs, state.n_frames, state.stft_flattened);

#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (backend && ggml_backend_is_cuda(backend) && !state.stft_flattened.empty()) {
        EnsureCudaHostRegistered(state.stft_flattened.data(),
                                 state.stft_flattened.size() * sizeof(float),
                                 cudaHostRegisterPortable | cudaHostRegisterReadOnly,
                                 state.cuda_reg_stft_ptr,
                                 state.cuda_reg_stft_bytes);
    }
#endif
}

std::shared_ptr<Inference::ChunkState> Inference::PreProcessChunk(std::vector<float> chunk_audio, int64_t id) {
    auto state = std::make_shared<ChunkState>();
    PreProcessChunkInto(*state, std::move(chunk_audio), id);

    return state;
}

void Inference::RunInference(std::shared_ptr<ChunkState> state) {
    if (!state || state->stft_flattened.empty()) return;

    ggml_backend_t backend = model_ ? model_->GetBackend() : nullptr;
    if (!backend) {
        throw std::runtime_error("Error: ggml backend not initialized");
    }

    const int timing_level = GetStreamTimingLevel();

    // 3. Ensure Graph
    if (!EnsureGraph(state->n_frames)) {
        return;
    }

    int n_bands = model_->GetNumBands();
    int n_frames = state->n_frames;

    const size_t input_bytes = ggml_nbytes(input_tensor_);
    const size_t mask_bytes = ggml_nbytes(mask_out_tensor_);

    const bool use_pinned_staging =
#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        (backend && ggml_backend_is_cuda(backend) && WantCudaPinnedStaging() &&
         cuda_pinned_in_ != nullptr && cuda_pinned_in_bytes_ >= input_bytes &&
         cuda_pinned_out_ != nullptr && cuda_pinned_out_bytes_ >= mask_bytes);
#else
        false;
#endif

    // Prepare position data
    // Use cached vectors to avoid allocation
    int required_time_size = n_frames * n_bands;
    if (pos_time_data_.size() != required_time_size) {
        pos_time_data_.resize(required_time_size);
        for(int i=0; i < required_time_size; ++i) pos_time_data_[i] = i % n_frames;
    }
    
    int required_freq_size = n_bands * n_frames;
    // Note: pos_freq logic (i % n_bands) depends on n_bands (constant) and total size.
    // If n_frames changes, size changes, and values might depend on n_frames?
    // Wait, pos_freq_data[i] = i % n_bands. 
    // This is valid regardless of n_frames as long as size is correct.
    // But we should regenerate if size changes.
    if (pos_freq_data_.size() != required_freq_size) {
        pos_freq_data_.resize(required_freq_size);
        for(int i=0; i < required_freq_size; ++i) pos_freq_data_[i] = i % n_bands;
    }

    // If we have pinned staging for position tensors, keep it up-to-date.
    const void* pos_time_src = pos_time_data_.data();
    const void* pos_freq_src = pos_freq_data_.data();
#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (backend && ggml_backend_is_cuda(backend) && WantCudaPinnedStaging()) {
        const size_t pos_time_bytes = static_cast<size_t>(pos_time_data_.size()) * sizeof(int32_t);
        if (cuda_pinned_pos_time_ != nullptr && cuda_pinned_pos_time_bytes_ >= pos_time_bytes) {
            std::memcpy(cuda_pinned_pos_time_, pos_time_data_.data(), pos_time_bytes);
            pos_time_src = cuda_pinned_pos_time_;
        }
        const size_t pos_freq_bytes = static_cast<size_t>(pos_freq_data_.size()) * sizeof(int32_t);
        if (cuda_pinned_pos_freq_ != nullptr && cuda_pinned_pos_freq_bytes_ >= pos_freq_bytes) {
            std::memcpy(cuda_pinned_pos_freq_, pos_freq_data_.data(), pos_freq_bytes);
            pos_freq_src = cuda_pinned_pos_freq_;
        }
    }
#endif

    // 4. Host -> Device
    using clock = std::chrono::high_resolution_clock;

    auto upload_inputs = [&]() {
        const void* input_src = state->stft_flattened.data();
        if (use_pinned_staging) {
            std::memcpy(cuda_pinned_in_, input_src, input_bytes);
            input_src = cuda_pinned_in_;
        }

        ggml_backend_tensor_set_async(backend, input_tensor_, input_src, 0, input_bytes);

        // Always upload RoPE position tensors.
        //
        // On some backends (observed on CUDA), these input buffers can become stale/corrupted across successive
        // graph runs if we only upload them once per n_frames. Re-uploading is cheap (~hundreds of KB) and keeps
        // multi-chunk inference stable.
        ggml_backend_tensor_set_async(backend, pos_time_, pos_time_src, 0, ggml_nbytes(pos_time_));
        ggml_backend_tensor_set_async(backend, pos_freq_, pos_freq_src, 0, ggml_nbytes(pos_freq_));
        uploaded_pos_n_frames_ = n_frames;
    };

    auto prepare_mask_output = [&]() {
        const size_t required_elems = static_cast<size_t>(ggml_nelements(mask_out_tensor_));

#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (backend && ggml_backend_is_cuda(backend)) {
            // If a resize might reallocate, unregister first (must not free registered memory).
            if (state->cuda_reg_mask_ptr != nullptr && state->mask_output.capacity() < required_elems) {
                TryCudaHostUnregister(state->cuda_reg_mask_ptr);
                state->cuda_reg_mask_ptr = nullptr;
                state->cuda_reg_mask_bytes = 0;
            }
        }
#endif

        state->mask_output.resize(required_elems);

#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (backend && ggml_backend_is_cuda(backend) && !state->mask_output.empty()) {
            EnsureCudaHostRegistered(state->mask_output.data(),
                                     state->mask_output.size() * sizeof(float),
                                     cudaHostRegisterPortable,
                                     state->cuda_reg_mask_ptr,
                                     state->cuda_reg_mask_bytes);
        }
#endif
    };

    // Debug timing breakdown (adds extra syncs; only enable when investigating GPU bubbles).
    if (timing_level >= 2) {
        const auto t0 = clock::now();
        upload_inputs();
        ggml_backend_synchronize(backend);
        const auto t1 = clock::now();

        ggml_backend_graph_compute_async(backend, gf_);
        ggml_backend_synchronize(backend);
        const auto t2 = clock::now();

        prepare_mask_output();
        void* mask_dst = state->mask_output.data();
        if (use_pinned_staging) {
            mask_dst = cuda_pinned_out_;
        }

        ggml_backend_tensor_get_async(backend, mask_out_tensor_, mask_dst, 0, mask_bytes);
        ggml_backend_synchronize(backend);
        if (use_pinned_staging) {
            std::memcpy(state->mask_output.data(), cuda_pinned_out_, mask_bytes);
        }
        const auto t3 = clock::now();

        const auto h2d_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        const auto compute_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        const auto d2h_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << "[StreamTiming] inf_detail offset=" << state->id
            << " n_frames=" << state->n_frames
            << " h2d_ms=" << (static_cast<double>(h2d_us) / 1000.0)
            << " compute_ms=" << (static_cast<double>(compute_us) / 1000.0)
            << " d2h_ms=" << (static_cast<double>(d2h_us) / 1000.0);
        LogStreamTimingLine(oss.str());
    } else {
        upload_inputs();

        // 5. Compute
        ggml_backend_graph_compute_async(backend, gf_);

        // 6. Device -> Host
        // Avoid reallocation if size roughly matches?
        // ggml_nelements(mask_out_tensor_) is fixed for a given n_frames.
        // state->mask_output is a vector. resize handles it (no op if same size).
        prepare_mask_output();

        void* mask_dst = state->mask_output.data();
        if (use_pinned_staging) {
            mask_dst = cuda_pinned_out_;
        }

        ggml_backend_tensor_get_async(backend, mask_out_tensor_, mask_dst, 0, mask_bytes);

        // Ensure H2D copies, compute, and D2H copy completed before post-processing consumes mask_output.
        ggml_backend_synchronize(backend);

        if (use_pinned_staging) {
            std::memcpy(state->mask_output.data(), cuda_pinned_out_, mask_bytes);
        }
    }

    const int mask_stats_level = GetMaskStatsLevel();
    if (mask_stats_level > 0 && (mask_stats_level >= 2 || state->id == 0)) {
        const FloatStats in_stats = ComputeFloatStats(state->stft_flattened);
        const FloatStats out_stats = ComputeFloatStats(state->mask_output);

        std::cerr << "[MaskStats] offset=" << state->id << " stft_flattened: n=" << in_stats.count
                  << " mean_abs=" << in_stats.mean_abs
                  << " max_abs=" << in_stats.max_abs
                  << " min=" << in_stats.min
                  << " max=" << in_stats.max
                  << std::endl;
        std::cerr << "[MaskStats] offset=" << state->id << " mask_output:    n=" << out_stats.count
                  << " mean_abs=" << out_stats.mean_abs
                  << " max_abs=" << out_stats.max_abs
                  << " min=" << out_stats.min
                  << " max=" << out_stats.max
                  << std::endl;
    }
}

void Inference::PostProcessChunk(std::shared_ptr<ChunkState> state) {
    if (!state || state->mask_output.empty()) return;

    // 7. Post-Process & ISTFT
    PostProcessAndISTFT(state->mask_output, state->stft_outputs, state->n_frames, state->final_audio);

    // 8. Trim
    for (auto& stem_audio : state->final_audio) {
        if (stem_audio.size() > state->input_audio.size()) {
           stem_audio.resize(state->input_audio.size());
        } else if (stem_audio.size() < state->input_audio.size()) {
           stem_audio.resize(state->input_audio.size(), 0.0f);
        }
    }

    const int mask_stats_level = GetMaskStatsLevel();
    if (mask_stats_level > 0 && (mask_stats_level >= 2 || state->id == 0)) {
        for (size_t s = 0; s < state->final_audio.size(); ++s) {
            const FloatStats audio_stats = ComputeFloatStats(state->final_audio[s]);
            std::cerr << "[MaskStats] offset=" << state->id << " audio_stem[" << s << "]:  n=" << audio_stats.count
                      << " mean_abs=" << audio_stats.mean_abs
                      << " max_abs=" << audio_stats.max_abs
                      << " min=" << audio_stats.min
                      << " max=" << audio_stats.max
                      << std::endl;
        }
    }
}

std::vector<std::vector<float>> Inference::ProcessChunk(const std::vector<float>& chunk_audio) {
    // Serial fallback
    const int timing_level = GetStreamTimingLevel();
    if (timing_level <= 0) {
        auto state = PreProcessChunk(chunk_audio, 0);
        RunInference(state);
        PostProcessChunk(state);
        return state->final_audio;
    }

    using clock = std::chrono::high_resolution_clock;
    const int part_len = static_cast<int>(chunk_audio.size() / 2);

    const auto t0 = clock::now();
    auto state = PreProcessChunk(chunk_audio, 0);
    const auto t1 = clock::now();
    RunInference(state);
    const auto t2 = clock::now();
    PostProcessChunk(state);
    const auto t3 = clock::now();

    const auto pre_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const auto inf_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const auto post_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "[StreamTiming] pre  offset=0 part=" << part_len << " n_frames=" << state->n_frames
        << " ms=" << (static_cast<double>(pre_us) / 1000.0);
    LogStreamTimingLine(oss.str());

    oss.str("");
    oss.clear();
    oss << std::fixed << std::setprecision(2);
    oss << "[StreamTiming] inf  offset=0 part=" << part_len << " n_frames=" << state->n_frames
        << " ms=" << (static_cast<double>(inf_us) / 1000.0);
    LogStreamTimingLine(oss.str());

    oss.str("");
    oss.clear();
    oss << std::fixed << std::setprecision(2);
    oss << "[StreamTiming] post offset=0 part=" << part_len << " n_frames=" << state->n_frames
        << " ms=" << (static_cast<double>(post_us) / 1000.0);
    LogStreamTimingLine(oss.str());

    return state->final_audio;
}

// =================================================================================================
// Pipelined Overlap-Add Logic
// =================================================================================================

// =================================================================================================
// Thread Safe Queue
// =================================================================================================

template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue(size_t max_size) : max_size_(max_size), shutdown_(false) {}

    ~ThreadSafeQueue() {
        Shutdown();
    }

    void Push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_push_.wait(lock, [this] { return queue_.size() < max_size_ || shutdown_; });
        if (shutdown_) return;
        queue_.push(std::move(item));
        cv_pop_.notify_one();
    }

    bool Pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_pop_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
        if (queue_.empty() && shutdown_) return false;
        item = std::move(queue_.front());
        queue_.pop();
        cv_push_.notify_one();
        return true;
    }

    bool TryPop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        cv_push_.notify_one();
        return true;
    }

    void Shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        cv_push_.notify_all();
        cv_pop_.notify_all();
    }

private:
    std::queue<T> queue_;
    size_t max_size_;
    bool shutdown_;
    std::mutex mutex_;
    std::condition_variable cv_push_;
    std::condition_variable cv_pop_;
};

// =================================================================================================
// Pipelined Overlap-Add Logic
// =================================================================================================

// =================================================================================================
// Pipelined Overlap-Add Logic (Optimized 3-Stage)
// =================================================================================================

std::vector<std::vector<float>> Inference::ProcessOverlapAddPipelined(const std::vector<float>& input_audio, 
                                                         int chunk_size, 
                                                         int num_overlap,
                                                         std::function<void(float)> progress_callback,
                                                         CancelCallback cancel_callback) {
    if (input_audio.empty()) return {};
    if (input_audio.size() % 2 != 0) {
        throw std::runtime_error("Error: Input audio must be interleaved stereo (even number of samples).");
    }
    
    // Parameters matches Python demix_track
    int channels = 2; 
    int C = chunk_size;
    
    int step = chunk_size / num_overlap;
    int fade_size = chunk_size / 10;
    int border = chunk_size - step;
    
    int n_input_samples = input_audio.size() / channels;

    // 1. Pad Input
    bool do_pad = (n_input_samples > 2 * border) && (border > 0);
    int pad_l = do_pad ? border : 0;
    int pad_r = do_pad ? border : 0;
    int n_padded_samples = n_input_samples + pad_l + pad_r;
    
    std::vector<float> padded_input;
    
    if (do_pad) {
        padded_input.resize(n_padded_samples * channels);
        // Copy center
        for (int i = 0; i < n_input_samples; ++i) {
            padded_input[(pad_l + i) * channels + 0] = input_audio[i * channels + 0];
            padded_input[(pad_l + i) * channels + 1] = input_audio[i * channels + 1];
        }
        // Reflect Left
        for (int i = 0; i < pad_l; ++i) {
            int src_idx = 1 + i; 
            if (src_idx >= n_input_samples) src_idx = n_input_samples - 1;
            int dst_idx = pad_l - 1 - i;
            padded_input[dst_idx * channels + 0] = input_audio[src_idx * channels + 0];
            padded_input[dst_idx * channels + 1] = input_audio[src_idx * channels + 1];
        }
        // Reflect Right
        for (int i = 0; i < pad_r; ++i) {
            int src_idx = n_input_samples - 2 - i;
            if (src_idx < 0) src_idx = 0;
            int dst_idx = pad_l + n_input_samples + i;
            padded_input[dst_idx * channels + 0] = input_audio[src_idx * channels + 0];
            padded_input[dst_idx * channels + 1] = input_audio[src_idx * channels + 1];
        }
    } else {
        padded_input = input_audio;
    }

    std::vector<std::vector<float>> result; // [stems][samples]
    std::vector<float> counter(n_padded_samples * channels, 0.0f);
    std::vector<float> window_base = GetWindow(chunk_size, fade_size);
    std::mutex result_mutex; // Protects 'result' and 'counter'
    std::atomic<bool> cancel_requested{false};
    
    // lambda to extract chunk 'i'
    auto extract_chunk = [&](int i) -> std::vector<float> {
        if (i >= n_padded_samples) return {};
        
        int remaining = n_padded_samples - i;
        int part_len = std::min(C, remaining);
        
        std::vector<float> chunk_in(C * channels, 0.0f);
        
        // Copy part
        for (int k = 0; k < part_len; ++k) {
            chunk_in[k * channels + 0] = padded_input[(i + k) * channels + 0];
            chunk_in[k * channels + 1] = padded_input[(i + k) * channels + 1];
        }
        
        // Pad short chunk if needed
        if (part_len < C) {
             int pad_amount = C - part_len;
             if (part_len > C / 2 + 1) {
                 // Reflect pad right
                 for(int k=0; k<pad_amount; ++k) {
                     int src_idx = part_len - 2 - k;
                     if(src_idx < 0) src_idx = 0;
                     chunk_in[(part_len + k)*2+0] = chunk_in[src_idx*2+0];
                     chunk_in[(part_len + k)*2+1] = chunk_in[src_idx*2+1];
                 }
             }
        }
        return chunk_in;
    };

    // lambda to accumulate result 'state' at offset 'i'
    // Now protected by mutex
    auto accumulate_result = [&](std::shared_ptr<ChunkState> state, int64_t i) {
        if (!state) return;
        const std::vector<std::vector<float>>& chunk_out_stems = state->final_audio;
        if (chunk_out_stems.empty()) return;
        
        std::lock_guard<std::mutex> lock(result_mutex);

        // Lazy Initialize result
        if (result.empty()) {
            int num_stems = chunk_out_stems.size();
            result.resize(num_stems, std::vector<float>(n_padded_samples * channels, 0.0f));
        }

        const int i32 = static_cast<int>(i);
        int remaining = n_padded_samples - i32;
        int part_len = std::min(C, remaining); 

        std::vector<float> window = window_base; // Copy
        if (i32 == 0) {
            for(int k=0; k<fade_size; ++k) window[k] = 1.0f;
        } else if (i32 + step >= n_padded_samples) {
            for(int k=0; k<fade_size; ++k) window[C - 1 - k] = 1.0f;
        }
        
        int num_stems = result.size();
        for (int k = 0; k < part_len; ++k) {
            float w = window[k];
            int res_idx = (i32 + k) * channels;
            int chk_idx = k * channels;
            
            for (int s = 0; s < num_stems; ++s) {
                 if (s >= chunk_out_stems.size()) continue;
                 // result[s] is huge, but we access linearly in this block
                 result[s][res_idx + 0] += chunk_out_stems[s][chk_idx + 0] * w;
                 result[s][res_idx + 1] += chunk_out_stems[s][chk_idx + 1] * w;
            }
            
            // Counter is same for all stems, just update once
            counter[res_idx + 0] += w;
            counter[res_idx + 1] += w;
        }
    };

    // =================================================================================================
    // 3-Stage Pipeline
    // =================================================================================================
    
    // Queues
    // Bounded size to prevents running out of memory
    // 3 items buffer is enough to keep GPU busy
    ThreadSafeQueue<std::shared_ptr<ChunkState>> input_queue(3);
    ThreadSafeQueue<std::shared_ptr<ChunkState>> output_queue(3);
    std::mutex exception_mutex;
    std::exception_ptr pipeline_exception = nullptr;

    auto set_pipeline_exception = [&](std::exception_ptr eptr) {
        {
            std::lock_guard<std::mutex> lock(exception_mutex);
            if (!pipeline_exception) {
                pipeline_exception = eptr;
            }
        }
        cancel_requested.store(true, std::memory_order_release);
        input_queue.Shutdown();
        output_queue.Shutdown();
    };
    
    // Structure to hold chunk metadata together
    struct ChunkTask {
        int64_t offset;
        std::shared_ptr<ChunkState> state;
    };
    
    // 1. Preprocessor Thread
    auto preproccessor = std::thread([&]() {
        try {
            int current_offset = 0;
            while (current_offset < n_padded_samples && !cancel_requested.load(std::memory_order_acquire)) {
                std::vector<float> chunk = extract_chunk(current_offset);
                
                auto state = PreProcessChunk(std::move(chunk), static_cast<int64_t>(current_offset));
                
                input_queue.Push(state);
                if (cancel_requested.load(std::memory_order_acquire)) {
                    break;
                }
                current_offset += step;
            }
        } catch (...) {
            set_pipeline_exception(std::current_exception());
        }
        input_queue.Shutdown();
    });
    
    // 3. Postprocessor Thread
    auto postprocessor = std::thread([&]() {
        try {
            std::shared_ptr<ChunkState> state;
            while (!cancel_requested.load(std::memory_order_acquire) && output_queue.Pop(state)) {
                // This does ISTFT (CPU intensive)
                PostProcessChunk(state);
                if (cancel_requested.load(std::memory_order_acquire)) {
                    break;
                }
                
                // Accumulate (Memory bandwidth intensive + Mutex)
                accumulate_result(state, state->id); // state->id holds offset
                
                if (!cancel_requested.load(std::memory_order_acquire) && progress_callback) {
                    float progress = (float)std::min<int64_t>(state->id + static_cast<int64_t>(step),
                                                             static_cast<int64_t>(n_padded_samples)) /
                                     static_cast<float>(n_padded_samples);
                    progress_callback(progress);
                }
            }
        } catch (...) {
            set_pipeline_exception(std::current_exception());
        }
    });
    
    auto poll_cancel_requested = [&]() -> bool {
        if (cancel_requested.load(std::memory_order_acquire)) {
            return true;
        }
        if (cancel_callback && cancel_callback()) {
            cancel_requested.store(true, std::memory_order_release);
            return true;
        }
        return false;
    };

    // 2. Main Thread (Inference Loop)
    bool cancelled = false;
    std::shared_ptr<ChunkState> state;
    try {
        while (true) {
            if (poll_cancel_requested()) {
                cancelled = true;
                break;
            }

            bool ok = input_queue.Pop(state);
            if (!ok) break; // Input queue shutdown and empty

            if (poll_cancel_requested()) {
                cancelled = true;
                break;
            }
            
            // This does GGML Inference (GPU intensive, Blocking)
            RunInference(state);

            if (poll_cancel_requested()) {
                cancelled = true;
                break;
            }
            
            output_queue.Push(state);
        }
    } catch (...) {
        set_pipeline_exception(std::current_exception());
    }
    
    if (cancelled) {
        cancel_requested.store(true, std::memory_order_release);
        input_queue.Shutdown();
    }

    // Wait for threads
    output_queue.Shutdown();
    if (preproccessor.joinable()) preproccessor.join();
    if (postprocessor.joinable()) postprocessor.join();

    if (pipeline_exception) {
        std::rethrow_exception(pipeline_exception);
    }

    if (cancel_requested.load(std::memory_order_acquire)) {
        throw std::runtime_error(kInferenceCancelledMessage);
    }

    // Normalize and Crop (in-place to avoid extra allocation)
    if (result.empty()) return {};

    int num_stems = result.size();

    // Normalize in-place
    for (int s = 0; s < num_stems; ++s) {
        for (int k = 0; k < n_input_samples; ++k) {
            int padded_idx = (pad_l + k) * channels;

            float w0 = counter[padded_idx + 0];
            float w1 = counter[padded_idx + 1];

            if (w0 < 1e-4f) w0 = 1.0f;
            if (w1 < 1e-4f) w1 = 1.0f;

            result[s][padded_idx + 0] /= w0;
            result[s][padded_idx + 1] /= w1;
        }
    }

    // Crop padding by resizing (no extra allocation)
    for (int s = 0; s < num_stems; ++s) {
        // Move unpadded data to front
        for (int k = 0; k < n_input_samples; ++k) {
            int src_idx = (pad_l + k) * channels;
            int dst_idx = k * channels;
            result[s][dst_idx + 0] = result[s][src_idx + 0];
            result[s][dst_idx + 1] = result[s][src_idx + 1];
        }
        result[s].resize(n_input_samples * channels);
    }

    return result;
}

std::vector<std::vector<float>> Inference::ProcessOverlapAdd(const std::vector<float>& input_audio, 
                                                int chunk_size, 
                                                int num_overlap,
                                                ModelCallback model_func,
                                                std::function<void(float)> progress_callback,
                                                CancelCallback cancel_callback) {
    if (input_audio.empty()) return {};
    if (input_audio.size() % 2 != 0) {
        throw std::runtime_error("Error: Input audio must be interleaved stereo (even number of samples).");
    }
    
    // Parameters matches Python demix_track
    int channels = 2; 
    int C = chunk_size;
    
    int step = chunk_size / num_overlap;
    int fade_size = chunk_size / 10;
    int border = chunk_size - step;
    
    int n_input_samples = input_audio.size() / channels;

    // 1. Pad Input
    bool do_pad = (n_input_samples > 2 * border) && (border > 0);
    int pad_l = do_pad ? border : 0;
    int pad_r = do_pad ? border : 0;
    int n_padded_samples = n_input_samples + pad_l + pad_r;
    
    std::vector<float> padded_input;
    
    if (do_pad) {
        padded_input.resize(n_padded_samples * channels);
        
        // Copy center
        for (int i = 0; i < n_input_samples; ++i) {
            padded_input[(pad_l + i) * channels + 0] = input_audio[i * channels + 0];
            padded_input[(pad_l + i) * channels + 1] = input_audio[i * channels + 1];
        }
        // Reflect Left
        for (int i = 0; i < pad_l; ++i) {
            int src_idx = 1 + i; 
            if (src_idx >= n_input_samples) src_idx = n_input_samples - 1;
            int dst_idx = pad_l - 1 - i;
            padded_input[dst_idx * channels + 0] = input_audio[src_idx * channels + 0];
            padded_input[dst_idx * channels + 1] = input_audio[src_idx * channels + 1];
        }
        // Reflect Right
        for (int i = 0; i < pad_r; ++i) {
            int src_idx = n_input_samples - 2 - i;
            if (src_idx < 0) src_idx = 0;
            int dst_idx = pad_l + n_input_samples + i;
            padded_input[dst_idx * channels + 0] = input_audio[src_idx * channels + 0];
            padded_input[dst_idx * channels + 1] = input_audio[src_idx * channels + 1];
        }
    } else {
        padded_input = input_audio;
    }

    std::vector<std::vector<float>> result; // [stems][samples]
    std::vector<float> counter(n_padded_samples * channels, 0.0f);
    std::vector<float> window_base = GetWindow(chunk_size, fade_size);
    
    int i = 0;
    int total_length = n_padded_samples;
    
    while (i < total_length) {
        if (cancel_callback && cancel_callback()) {
            throw std::runtime_error(kInferenceCancelledMessage);
        }

        int remaining = total_length - i;
        int part_len = std::min(C, remaining); // Logic matches Python slice [i:i+C]
        
        std::vector<float> chunk_in(C * channels, 0.0f);
        
        // Copy part
        for (int k = 0; k < part_len; ++k) {
            chunk_in[k * channels + 0] = padded_input[(i + k) * channels + 0];
            chunk_in[k * channels + 1] = padded_input[(i + k) * channels + 1];
        }
        
        // Pad short chunk if needed
        if (part_len < C) {
             int pad_amount = C - part_len;
             if (part_len > C / 2 + 1) {
                 // Reflect pad right
                 for(int k=0; k<pad_amount; ++k) {
                     int src_idx = part_len - 2 - k;
                     if(src_idx < 0) src_idx = 0;
                     chunk_in[(part_len + k)*2+0] = chunk_in[src_idx*2+0];
                     chunk_in[(part_len + k)*2+1] = chunk_in[src_idx*2+1];
                 }
             }
        }
        
        std::vector<std::vector<float>> chunk_out_stems = model_func(chunk_in);
        if (chunk_out_stems.empty()) {
             // ?
        }
        
        // Lazy Initialize result
        if (result.empty()) {
            int num_stems = chunk_out_stems.size();
            result.resize(num_stems, std::vector<float>(n_padded_samples * channels, 0.0f));
        }

        // Window Adjustment
        std::vector<float> window = window_base; // Copy
        if (i == 0) {
            for(int k=0; k<fade_size; ++k) window[k] = 1.0f;
        } else if (i + step >= total_length) {
            for(int k=0; k<fade_size; ++k) window[C - 1 - k] = 1.0f;
        }
        
        // Accumulate
        int num_stems = result.size();
        for (int k = 0; k < part_len; ++k) {
            float w = window[k];
            int res_idx = (i + k) * channels;
            int chk_idx = k * channels;
            
            for (int s = 0; s < num_stems; ++s) {
                 if (s >= chunk_out_stems.size()) continue;
                 const auto& stem_chunk = chunk_out_stems[s];
                 result[s][res_idx + 0] += stem_chunk[chk_idx + 0] * w;
                 result[s][res_idx + 1] += stem_chunk[chk_idx + 1] * w;
            }
            
            counter[res_idx + 0] += w;
            counter[res_idx + 1] += w;
        }
        
        i += step;
        if (progress_callback) {
             float progress = (float)std::min(i, total_length) / total_length;
             progress_callback(progress);
        }
    }
    
    // Normalize and Crop
    if (result.empty()) return {};

    int num_stems = result.size();
    std::vector<std::vector<float>> final_output_stems(num_stems);
    
    for (int s = 0; s < num_stems; ++s) {
        final_output_stems[s].resize(n_input_samples * channels);
        for (int k = 0; k < n_input_samples; ++k) {
            int padded_idx = (pad_l + k) * channels;
            int final_idx = k * channels;
            
            float w0 = counter[padded_idx + 0];
            float w1 = counter[padded_idx + 1];
            
            if (w0 < 1e-4f) w0 = 1.0f;
            if (w1 < 1e-4f) w1 = 1.0f;
            
            final_output_stems[s][final_idx + 0] = result[s][padded_idx + 0] / w0;
            final_output_stems[s][final_idx + 1] = result[s][padded_idx + 1] / w1;
        }
    }
    
    return final_output_stems;
}

// =================================================================================================
// Streaming API (constant-memory overlap-add)
// =================================================================================================

Inference::OverlapAddStreamer::OverlapAddStreamer(int chunk_size, int num_overlap, int num_stems, ModelCallback model_func)
    : chunk_size_(chunk_size),
      num_overlap_(num_overlap),
      num_stems_(num_stems),
      model_func_(std::move(model_func)) {
    if (chunk_size_ <= 0) {
        throw std::runtime_error("Error: chunk_size must be a positive integer");
    }
    if (num_overlap_ < 1) {
        throw std::runtime_error("Error: num_overlap must be at least 1");
    }
    if (num_stems_ <= 0) {
        throw std::runtime_error("Error: num_stems must be a positive integer");
    }

    step_ = chunk_size_ / num_overlap_;
    if (step_ <= 0) {
        throw std::runtime_error("Error: invalid overlap configuration (chunk_size / num_overlap == 0)");
    }
    border_ = chunk_size_ - step_;
    fade_size_ = chunk_size_ / 10;

    accum_.assign(num_stems_, std::vector<float>(chunk_size_ * kChannels, 0.0f));
    counter_.assign(chunk_size_, 0.0f);
    window_base_ = GetWindow(chunk_size_, fade_size_);
}

void Inference::OverlapAddStreamer::Feed(const std::vector<float>& input_audio) {
    if (finalized_) {
        throw std::runtime_error("Error: streaming context already finalized");
    }
    if (input_finalized_) {
        throw std::runtime_error("Error: streaming input already finalized");
    }
    if (input_audio.empty()) return;
    if (input_audio.size() % kChannels != 0) {
        throw std::runtime_error("Error: input audio must be interleaved stereo (even number of floats).");
    }

    const int64_t in_frames = static_cast<int64_t>(input_audio.size() / kChannels);
    total_input_samples_ += in_frames;

    // Update tail buffer for right reflection padding (keep last border_+2 frames).
    const int64_t tail_max_frames = std::max<int64_t>(static_cast<int64_t>(border_) + 2, 2);
    if (in_frames >= tail_max_frames) {
        tail_buffer_.assign(input_audio.end() - static_cast<std::ptrdiff_t>(tail_max_frames * kChannels), input_audio.end());
    } else {
        const int64_t old_frames = static_cast<int64_t>(tail_buffer_.size() / kChannels);
        if (old_frames + in_frames <= tail_max_frames) {
            tail_buffer_.insert(tail_buffer_.end(), input_audio.begin(), input_audio.end());
        } else {
            const int64_t keep_old = tail_max_frames - in_frames;
            std::vector<float> new_tail;
            new_tail.reserve(static_cast<size_t>(tail_max_frames) * kChannels);

            if (keep_old > 0 && old_frames > 0) {
                const int64_t copy_old = std::min<int64_t>(keep_old, old_frames);
                const size_t start_old = static_cast<size_t>(old_frames - copy_old) * kChannels;
                new_tail.insert(new_tail.end(), tail_buffer_.begin() + static_cast<std::ptrdiff_t>(start_old), tail_buffer_.end());
            }
            new_tail.insert(new_tail.end(), input_audio.begin(), input_audio.end());
            tail_buffer_ = std::move(new_tail);
        }
    }

    if (!decided_pad_) {
        prebuffer_.insert(prebuffer_.end(), input_audio.begin(), input_audio.end());
        DecidePaddingIfPossible(/*flushing=*/false);
        if (!decided_pad_) {
            // Need more input to decide do_pad (short files). Buffer until FinalizeInput().
            return;
        }
    } else {
        AppendToInputBuffer(input_audio);
    }
}

void Inference::OverlapAddStreamer::FinalizeInput() {
    if (finalized_) {
        throw std::runtime_error("Error: streaming context already finalized");
    }
    if (input_finalized_) return;
    input_finalized_ = true;

    DecidePaddingIfPossible(/*flushing=*/true);
    if (!decided_pad_) {
        // Should not happen, but be safe.
        total_length_padded_ = 0;
        return;
    }

    if (do_pad_) {
        AppendToInputBuffer(BuildRightPad());
    }

    const int64_t pad_r = do_pad_ ? border_ : 0;
    total_length_padded_ = total_input_samples_ + crop_left_ + pad_r;
}

bool Inference::OverlapAddStreamer::TryScheduleNext(ScheduledChunk& out) {
    out = ScheduledChunk{};

    if (!decided_pad_) {
        // Not enough input to decide do_pad yet.
        return false;
    }

    if (input_finalized_) {
        if (schedule_offset_ >= total_length_padded_) return false;
        auto ex = ExtractChunkAtOffset(schedule_offset_, total_length_padded_, /*flushing=*/true);
        out.offset = schedule_offset_;
        out.part_len = ex.part_len;
        out.chunk_in = std::move(ex.chunk_in);
        out.is_first = (out.offset == 0);
        out.is_last = (out.offset + step_ >= total_length_padded_);
        schedule_offset_ += step_;
        return true;
    }

    const int64_t available_padded = AvailablePaddedSamples();
    if (schedule_offset_ + chunk_size_ > available_padded) return false;

    auto ex = ExtractChunkAtOffset(schedule_offset_, available_padded, /*flushing=*/false);
    out.offset = schedule_offset_;
    out.part_len = ex.part_len;
    out.chunk_in = std::move(ex.chunk_in);
    out.is_first = (out.offset == 0);
    out.is_last = false;
    schedule_offset_ += step_;
    return true;
}

bool Inference::OverlapAddStreamer::TryScheduleNextMeta(ScheduledChunk& out) {
    out = ScheduledChunk{};

    if (!decided_pad_) {
        // Not enough input to decide do_pad yet.
        return false;
    }

    if (input_finalized_) {
        if (schedule_offset_ >= total_length_padded_) return false;

        const int64_t remaining = total_length_padded_ - schedule_offset_;
        const int part_len = static_cast<int>(std::min<int64_t>(chunk_size_, remaining));
        if (part_len <= 0) return false;

        out.offset = schedule_offset_;
        out.part_len = part_len;
        out.is_first = (out.offset == 0);
        out.is_last = (out.offset + step_ >= total_length_padded_);
        schedule_offset_ += step_;
        return true;
    }

    const int64_t available_padded = AvailablePaddedSamples();
    if (schedule_offset_ + chunk_size_ > available_padded) return false;

    out.offset = schedule_offset_;
    out.part_len = chunk_size_;
    out.is_first = (out.offset == 0);
    out.is_last = false;
    schedule_offset_ += step_;
    return true;
}

void Inference::OverlapAddStreamer::MaterializeChunkInput(const ScheduledChunk& scheduled, std::vector<float>& dst) const {
    if (scheduled.part_len <= 0) {
        dst.clear();
        return;
    }

    const size_t total_floats = static_cast<size_t>(chunk_size_) * kChannels;
    dst.resize(total_floats);
    std::fill(dst.begin(), dst.end(), 0.0f);

    const int64_t rel_samples = scheduled.offset - input_base_;
    if (rel_samples < 0) {
        throw std::runtime_error("Internal error: scheduled.offset < input_base_ in MaterializeChunkInput");
    }

    const size_t start_float = input_offset_ + static_cast<size_t>(rel_samples) * kChannels;
    const size_t copy_floats = static_cast<size_t>(scheduled.part_len) * kChannels;
    if (start_float + copy_floats > input_buffer_.size()) {
        throw std::runtime_error("Internal error: input buffer underrun in MaterializeChunkInput");
    }

    std::memcpy(dst.data(), input_buffer_.data() + start_float, copy_floats * sizeof(float));

    // Pad short chunk if needed: matches batch ProcessOverlapAdd.
    if (scheduled.part_len < chunk_size_) {
        int pad_amount = chunk_size_ - scheduled.part_len;
        if (scheduled.part_len > chunk_size_ / 2 + 1) {
            for (int k = 0; k < pad_amount; ++k) {
                int src_idx = scheduled.part_len - 2 - k;
                if (src_idx < 0) src_idx = 0;
                dst[static_cast<size_t>(scheduled.part_len + k) * kChannels + 0] =
                    dst[static_cast<size_t>(src_idx) * kChannels + 0];
                dst[static_cast<size_t>(scheduled.part_len + k) * kChannels + 1] =
                    dst[static_cast<size_t>(src_idx) * kChannels + 1];
            }
        }
    }
}

void Inference::OverlapAddStreamer::ConsumeScheduledAppend(const ScheduledChunk& scheduled,
                                                          const std::vector<std::vector<float>>& chunk_out,
                                                          std::vector<std::vector<float>>& out_acc) {
    if (scheduled.offset != current_offset_) {
        throw std::runtime_error("Error: ConsumeScheduled() called out of order (offset mismatch)");
    }

    AccumulateChunk(chunk_out, scheduled.part_len, scheduled.is_first, scheduled.is_last);

    int ready_samples = step_;
    int64_t remaining_limit = -1;
    if (input_finalized_) {
        const int64_t remaining_padded = total_length_padded_ - produced_padded_samples_;
        ready_samples = static_cast<int>(std::min<int64_t>(step_, std::max<int64_t>(0, remaining_padded)));

        remaining_limit = std::max<int64_t>(0, total_input_samples_ - emitted_output_samples_);
    }

    AppendReadyOutput(ready_samples, remaining_limit, out_acc);

    ShiftWindowByStep();
    current_offset_ += step_;
    produced_padded_samples_ += step_;
    DropInputPrefixUpTo(current_offset_);
}

std::vector<std::vector<float>> Inference::OverlapAddStreamer::ConsumeScheduled(const ScheduledChunk& scheduled,
                                                                                const std::vector<std::vector<float>>& chunk_out) {
    std::vector<std::vector<float>> ready;
    ConsumeScheduledAppend(scheduled, chunk_out, ready);
    return ready;
}

void Inference::OverlapAddStreamer::AppendToInputBuffer(const std::vector<float>& interleaved) {
    if (interleaved.empty()) return;
    input_buffer_.insert(input_buffer_.end(), interleaved.begin(), interleaved.end());
}

int64_t Inference::OverlapAddStreamer::AvailablePaddedSamples() const {
    if (input_offset_ >= input_buffer_.size()) {
        return input_base_;
    }
    return input_base_ + static_cast<int64_t>((input_buffer_.size() - input_offset_) / kChannels);
}

void Inference::OverlapAddStreamer::ReadFromInputBuffer(int64_t abs_sample_idx,
                                                        int part_len,
                                                        std::vector<float>& dst) const {
    if (part_len <= 0) {
        dst.clear();
        return;
    }

    const int64_t rel_samples = abs_sample_idx - input_base_;
    if (rel_samples < 0) {
        throw std::runtime_error("Internal error: abs_sample_idx < input_base_ in ReadFromInputBuffer");
    }

    const size_t start_float = input_offset_ + static_cast<size_t>(rel_samples) * kChannels;
    const size_t copy_floats = static_cast<size_t>(part_len) * kChannels;
    if (start_float + copy_floats > input_buffer_.size()) {
        throw std::runtime_error("Internal error: input buffer underrun in ReadFromInputBuffer");
    }

    dst.resize(copy_floats);
    std::memcpy(dst.data(), input_buffer_.data() + start_float, copy_floats * sizeof(float));
}

void Inference::OverlapAddStreamer::DropInputPrefixUpTo(int64_t new_base_sample) {
    if (new_base_sample <= input_base_) return;

    int64_t drop_samples = new_base_sample - input_base_;
    size_t drop_floats = static_cast<size_t>(drop_samples) * kChannels;

    input_base_ = new_base_sample;
    input_offset_ += drop_floats;

    if (input_offset_ >= input_buffer_.size()) {
        input_buffer_.clear();
        input_offset_ = 0;
        return;
    }

    // Periodically compact to keep memory bounded.
    // This is O(n), so don't do it on every step.
    if (input_offset_ > (1u << 20)) { // ~1MB of floats
        input_buffer_.erase(input_buffer_.begin(), input_buffer_.begin() + static_cast<std::ptrdiff_t>(input_offset_));
        input_offset_ = 0;
    }
}

void Inference::OverlapAddStreamer::MaterializeLeftPadAndMovePrebuffer() {
    crop_left_ = do_pad_ ? border_ : 0;

    // Reset stream cursor state
    current_offset_ = 0;
    schedule_offset_ = 0;
    produced_padded_samples_ = 0;
    emitted_output_samples_ = 0;

    input_base_ = 0;
    input_offset_ = 0;
    input_buffer_.clear();

    if (prebuffer_.empty()) {
        prebuffer_.shrink_to_fit();
        return;
    }

    if (do_pad_) {
        const int64_t n_input_samples_now = static_cast<int64_t>(prebuffer_.size() / kChannels);
        std::vector<float> left_pad(static_cast<size_t>(border_) * kChannels, 0.0f);

        // Reflect Left: matches ProcessOverlapAdd exactly.
        for (int i = 0; i < border_; ++i) {
            int64_t src_idx = 1 + static_cast<int64_t>(i);
            if (src_idx >= n_input_samples_now) src_idx = n_input_samples_now - 1;
            int64_t dst_idx = static_cast<int64_t>(border_ - 1 - i);
            left_pad[static_cast<size_t>(dst_idx) * kChannels + 0] =
                prebuffer_[static_cast<size_t>(src_idx) * kChannels + 0];
            left_pad[static_cast<size_t>(dst_idx) * kChannels + 1] =
                prebuffer_[static_cast<size_t>(src_idx) * kChannels + 1];
        }

        input_buffer_ = std::move(left_pad);
        input_buffer_.insert(input_buffer_.end(), prebuffer_.begin(), prebuffer_.end());
    } else {
        input_buffer_ = std::move(prebuffer_);
    }

    prebuffer_.clear();
    prebuffer_.shrink_to_fit();
}

void Inference::OverlapAddStreamer::DecidePaddingIfPossible(bool flushing) {
    if (decided_pad_) return;

    if (border_ <= 0) {
        // Matches batch: do_pad requires border > 0
        do_pad_ = false;
        decided_pad_ = true;
        MaterializeLeftPadAndMovePrebuffer();
        return;
    }

    // We can decide do_pad=true as soon as total_input_samples_ exceeds threshold.
    // We cannot decide do_pad=false until flushing (end of stream).
    if (!flushing && total_input_samples_ <= 2LL * border_) {
        return;
    }

    do_pad_ = (total_input_samples_ > 2LL * border_) && (border_ > 0);
    decided_pad_ = true;
    MaterializeLeftPadAndMovePrebuffer();
}

Inference::OverlapAddStreamer::ChunkExtractResult Inference::OverlapAddStreamer::ExtractChunkAtOffset(
    int64_t abs_offset,
    int64_t total_length_padded,
    bool /*flushing*/) const {
    ChunkExtractResult r;
    if (abs_offset >= total_length_padded) return r;

    int64_t remaining = total_length_padded - abs_offset;
    int part_len = static_cast<int>(std::min<int64_t>(chunk_size_, remaining));
    if (part_len <= 0) return r;

    r.part_len = part_len;
    r.chunk_in.assign(static_cast<size_t>(chunk_size_) * kChannels, 0.0f);

    // Copy available samples from input buffer
    int64_t rel_samples = abs_offset - input_base_;
    if (rel_samples < 0) {
        throw std::runtime_error("Internal error: abs_offset < input_base_ in ExtractChunkAtOffset");
    }

    size_t start_float = input_offset_ + static_cast<size_t>(rel_samples) * kChannels;
    size_t copy_floats = static_cast<size_t>(part_len) * kChannels;

    if (start_float + copy_floats > input_buffer_.size()) {
        throw std::runtime_error("Internal error: input buffer underrun in ExtractChunkAtOffset");
    }

    std::memcpy(r.chunk_in.data(), input_buffer_.data() + start_float, copy_floats * sizeof(float));

    // Pad short chunk if needed: matches batch ProcessOverlapAdd.
    if (part_len < chunk_size_) {
        int pad_amount = chunk_size_ - part_len;
        if (part_len > chunk_size_ / 2 + 1) {
            for (int k = 0; k < pad_amount; ++k) {
                int src_idx = part_len - 2 - k;
                if (src_idx < 0) src_idx = 0;
                r.chunk_in[static_cast<size_t>(part_len + k) * kChannels + 0] =
                    r.chunk_in[static_cast<size_t>(src_idx) * kChannels + 0];
                r.chunk_in[static_cast<size_t>(part_len + k) * kChannels + 1] =
                    r.chunk_in[static_cast<size_t>(src_idx) * kChannels + 1];
            }
        }
    }

    return r;
}

void Inference::OverlapAddStreamer::AccumulateChunk(const std::vector<std::vector<float>>& chunk_out,
                                                    int part_len,
                                                    bool is_first,
                                                    bool is_last) {
    if (part_len <= 0) return;

    for (int k = 0; k < part_len; ++k) {
        float w = window_base_[k];
        if (is_first && k < fade_size_) {
            w = 1.0f;
        }
        if (is_last && k >= chunk_size_ - fade_size_) {
            w = 1.0f;
        }
        size_t chk_idx = static_cast<size_t>(k) * kChannels;

        for (int s = 0; s < num_stems_; ++s) {
            if (s >= (int)chunk_out.size()) continue;
            const auto& stem_chunk = chunk_out[s];
            accum_[s][chk_idx + 0] += stem_chunk[chk_idx + 0] * w;
            accum_[s][chk_idx + 1] += stem_chunk[chk_idx + 1] * w;
        }

        counter_[k] += w;
    }
}

void Inference::OverlapAddStreamer::AppendReadyOutput(int ready_samples,
                                                      int64_t remaining_output_limit,
                                                      std::vector<std::vector<float>>& out_acc) {
    if (ready_samples <= 0) return;

    const int64_t first_abs = produced_padded_samples_;
    const int64_t last_abs = produced_padded_samples_ + ready_samples;

    int64_t out_start = std::max<int64_t>(0, first_abs - crop_left_);
    int64_t out_end = std::min<int64_t>(total_input_samples_, last_abs - crop_left_);

    if (remaining_output_limit >= 0) {
        out_end = std::min<int64_t>(out_end, out_start + remaining_output_limit);
    }

    if (out_end <= out_start) return;

    const int64_t out_count = out_end - out_start;
    const int local_start = static_cast<int>(out_start + crop_left_ - first_abs);

    if (out_acc.size() != static_cast<size_t>(num_stems_)) {
        out_acc.resize(static_cast<size_t>(num_stems_));
    }

    const size_t append_floats = static_cast<size_t>(out_count) * kChannels;
    std::array<size_t, 8> base_small{};
    std::vector<size_t> base_big;
    size_t* base = nullptr;
    if (num_stems_ <= static_cast<int>(base_small.size())) {
        base = base_small.data();
    } else {
        base_big.resize(static_cast<size_t>(num_stems_), 0);
        base = base_big.data();
    }
    for (int s = 0; s < num_stems_; ++s) {
        base[s] = out_acc[static_cast<size_t>(s)].size();
        out_acc[static_cast<size_t>(s)].resize(base[s] + append_floats);
    }

    for (int64_t i = 0; i < out_count; ++i) {
        const int k = local_start + static_cast<int>(i);
        float w = counter_[k];
        if (w < 1e-4f) w = 1.0f;

        size_t src_idx = static_cast<size_t>(k) * kChannels;
        size_t dst_off = static_cast<size_t>(i) * kChannels;

        for (int s = 0; s < num_stems_; ++s) {
            auto& dst = out_acc[static_cast<size_t>(s)];
            const size_t dst_idx = base[s] + dst_off;
            dst[dst_idx + 0] = accum_[s][src_idx + 0] / w;
            dst[dst_idx + 1] = accum_[s][src_idx + 1] / w;
        }
    }

    emitted_output_samples_ += out_count;
}

std::vector<std::vector<float>> Inference::OverlapAddStreamer::EmitReadyOutput(int ready_samples,
                                                                               int64_t remaining_output_limit) {
    std::vector<std::vector<float>> out;
    AppendReadyOutput(ready_samples, remaining_output_limit, out);
    return out;
}

void Inference::OverlapAddStreamer::ShiftWindowByStep() {
    const int keep = chunk_size_ - step_;

    for (int s = 0; s < num_stems_; ++s) {
        auto& a = accum_[s];
        if (keep > 0) {
            std::memmove(a.data(), a.data() + static_cast<size_t>(step_) * kChannels,
                         static_cast<size_t>(keep) * kChannels * sizeof(float));
        }
        std::fill(a.begin() + static_cast<std::ptrdiff_t>(keep) * kChannels, a.end(), 0.0f);
    }

    if (keep > 0) {
        std::memmove(counter_.data(), counter_.data() + step_, static_cast<size_t>(keep) * sizeof(float));
    }
    std::fill(counter_.begin() + keep, counter_.end(), 0.0f);
}

std::vector<float> Inference::OverlapAddStreamer::BuildRightPad() const {
    if (border_ <= 0) return {};

    std::vector<float> right_pad(static_cast<size_t>(border_) * kChannels, 0.0f);

    const int64_t tail_frames = static_cast<int64_t>(tail_buffer_.size() / kChannels);
    const int64_t tail_start = total_input_samples_ - tail_frames;

    for (int i = 0; i < border_; ++i) {
        int64_t src_idx = total_input_samples_ - 2 - static_cast<int64_t>(i);
        if (src_idx < 0) src_idx = 0;
        int64_t off = src_idx - tail_start;
        if (off < 0) off = 0;
        if (off >= tail_frames) off = tail_frames - 1;

        right_pad[static_cast<size_t>(i) * kChannels + 0] = tail_buffer_[static_cast<size_t>(off) * kChannels + 0];
        right_pad[static_cast<size_t>(i) * kChannels + 1] = tail_buffer_[static_cast<size_t>(off) * kChannels + 1];
    }

    return right_pad;
}

std::vector<std::vector<float>> Inference::OverlapAddStreamer::Push(const std::vector<float>& input_audio) {
    if (finalized_) {
        throw std::runtime_error("Error: streaming context already finalized");
    }
    if (!model_func_) {
        throw std::runtime_error("Error: Push() requires a model_func callback (construct with a non-null callback)");
    }

    Feed(input_audio);

    std::vector<std::vector<float>> out_acc;
    ScheduledChunk scheduled;
    while (TryScheduleNext(scheduled)) {
        auto chunk_out = model_func_(scheduled.chunk_in);
        ConsumeScheduledAppend(scheduled, chunk_out, out_acc);
    }

    return out_acc;
}

std::vector<std::vector<float>> Inference::OverlapAddStreamer::Finalize() {
    if (finalized_) {
        throw std::runtime_error("Error: streaming context already finalized");
    }
    if (!model_func_) {
        throw std::runtime_error("Error: Finalize() requires a model_func callback (construct with a non-null callback)");
    }
    FinalizeInput();
    finalized_ = true;

    std::vector<std::vector<float>> out_acc;
    ScheduledChunk scheduled;
    while (TryScheduleNext(scheduled)) {
        auto chunk_out = model_func_(scheduled.chunk_in);
        ConsumeScheduledAppend(scheduled, chunk_out, out_acc);

        if (emitted_output_samples_ >= total_input_samples_) {
            break;
        }
    }

    return out_acc;
}

static size_t GetStreamPipelineDepth() {
    const char* env = std::getenv("BSR_STREAM_PIPELINE_DEPTH");
    // Default to depth=1 to keep host-side in-flight buffers minimal for very long audio.
    // Users can raise this to 2-8 to overlap pre/inf/post stages for higher throughput.
    if (!env || !*env) return 1;

    char* end = nullptr;
    long v = std::strtol(env, &end, 10);
    if (end == env) return 1;

    if (v < 1) v = 1;
    if (v > 8) v = 8;
    return static_cast<size_t>(v);
}

struct Inference::StreamImpl {
    struct Task {
        OverlapAddStreamer::ScheduledChunk scheduled;
        std::shared_ptr<ChunkState> state;
    };

    explicit StreamImpl(Inference* engine, int chunk_size, int num_overlap, int num_stems, bool pipelined)
        : engine_(engine),
          pipelined_(pipelined),
          pipeline_depth_(GetStreamPipelineDepth()),
          num_stems_(num_stems),
          timing_level_(GetStreamTimingLevel()),
          mem_level_(GetStreamMemLevel()),
          chunk_size_(chunk_size),
          num_overlap_(num_overlap),
          step_((num_overlap > 0) ? (chunk_size / num_overlap) : 0) {
        if (!engine_) {
            throw std::runtime_error("Error: invalid Inference instance");
        }

        if (!pipelined_) {
            serial_ = std::make_unique<OverlapAddStreamer>(chunk_size, num_overlap, num_stems,
                                                           [this](const std::vector<float>& chunk) {
                                                               return engine_->ProcessChunk(chunk);
                                                           });
            return;
        }

        // Manual overlap-add scheduler/accumulator (model callback handled by pipeline threads)
        oa_ = std::make_unique<OverlapAddStreamer>(chunk_size, num_overlap, num_stems, nullptr);

        // Bound all queues by pipeline_depth_ so total in-flight memory stays constant.
        q_pre_ = std::make_unique<ThreadSafeQueue<Task>>(pipeline_depth_);
        q_inf_ = std::make_unique<ThreadSafeQueue<Task>>(pipeline_depth_);
        q_post_ = std::make_unique<ThreadSafeQueue<Task>>(pipeline_depth_);
        q_done_ = std::make_unique<ThreadSafeQueue<Task>>(pipeline_depth_);

        // Reuse ChunkState allocations to avoid per-chunk heap growth (large vectors like mask_output can fragment).
        q_free_state_ = std::make_unique<ThreadSafeQueue<std::shared_ptr<ChunkState>>>(pipeline_depth_);
        for (size_t i = 0; i < pipeline_depth_; ++i) {
            q_free_state_->Push(std::make_shared<ChunkState>());
        }

        StartThreads();
    }

    ~StreamImpl() {
        ShutdownAndJoin();
        PrintTimingSummaryIfEnabled();
    }

    void ProcessInto(const std::vector<float>& input_chunk, std::vector<std::vector<float>>& out_acc) {
        if (!pipelined_) {
            out_acc = serial_->Push(input_chunk);
            return;
        }

        MaybeRethrow();
        ResetOut(out_acc);

        // Drain any completed chunks first to avoid backpressure.
        DrainDone(out_acc);
        ScheduleMore();

        if (!input_chunk.empty()) {
            // Backpressure: prevent unbounded buffering of input audio inside the overlap-add scheduler.
            // If the pipeline is full, wait until at least one chunk completes before accepting more input.
            while (in_flight_ >= pipeline_depth_) {
                Task done;
                bool ok = q_done_->Pop(done);
                if (!ok) break;

                if (in_flight_ == 0) {
                    throw std::runtime_error("Internal error: in-flight underflow in streaming process");
                }
                --in_flight_;

                oa_->ConsumeScheduledAppend(done.scheduled, done.state->final_audio, out_acc);
                MaybeLogMem("wait", done);
                RecycleState(std::move(done.state));

                DrainDone(out_acc);
                MaybeRethrow();
            }

            oa_->Feed(input_chunk);
            ScheduleMore();
        }

        DrainDone(out_acc);
        MaybeRethrow();
    }

    std::vector<std::vector<float>> Process(const std::vector<float>& input_chunk) {
        std::vector<std::vector<float>> out_acc;
        ProcessInto(input_chunk, out_acc);
        return out_acc;
    }

    void FinalizeInto(std::vector<std::vector<float>>& out_acc) {
        if (!pipelined_) {
            out_acc = serial_->Finalize();
            return;
        }
        if (finalized_) {
            throw std::runtime_error("Error: stream already finalized");
        }
        finalized_ = true;

        MaybeRethrow();

        ResetOut(out_acc);
        DrainDone(out_acc);

        oa_->FinalizeInput();

        bool no_more_to_schedule = false;
        bool pre_shutdown = false;

        while (true) {
            MaybeRethrow();

            // Schedule more work if possible.
            if (!no_more_to_schedule) {
                while (in_flight_ < pipeline_depth_) {
                    OverlapAddStreamer::ScheduledChunk scheduled;
                    if (!oa_->TryScheduleNextMeta(scheduled)) {
                        no_more_to_schedule = true;
                        break;
                    }

                    Task task;
                    task.scheduled = std::move(scheduled);

                    std::shared_ptr<ChunkState> state;
                    if (q_free_state_) {
                        if (!q_free_state_->Pop(state)) {
                            no_more_to_schedule = true;
                            break;
                        }
                    }
                    if (!state) {
                        state = std::make_shared<ChunkState>();
                    }

                    state->id = task.scheduled.offset;
                    oa_->MaterializeChunkInput(task.scheduled, state->input_audio);
                    task.state = std::move(state);

                    q_pre_->Push(std::move(task));
                    ++in_flight_;
                }
            }

            if (no_more_to_schedule && !pre_shutdown) {
                q_pre_->Shutdown();
                pre_shutdown = true;
            }

            // Drain anything already done.
            DrainDone(out_acc);

            if (no_more_to_schedule && in_flight_ == 0) {
                break;
            }

            // Block until at least one chunk completes to make progress.
            Task done;
            bool ok = q_done_->Pop(done);
            if (!ok) {
                // Pipeline shut down (likely due to exception).
                break;
            }

            if (in_flight_ == 0) {
                throw std::runtime_error("Internal error: in-flight underflow in streaming finalize");
            }
            --in_flight_;

            oa_->ConsumeScheduledAppend(done.scheduled, done.state->final_audio, out_acc);
            MaybeLogMem("final", done);
            RecycleState(std::move(done.state));
        }

        ShutdownAndJoin();
        DrainDone(out_acc);
        MaybeRethrow();
    }

    std::vector<std::vector<float>> Finalize() {
        std::vector<std::vector<float>> out_acc;
        FinalizeInto(out_acc);
        return out_acc;
    }

private:
    Inference* engine_ = nullptr;
    bool pipelined_ = false;
    size_t pipeline_depth_ = 3;
    int num_stems_ = 0;
    int chunk_size_ = 0;
    int num_overlap_ = 0;
    int step_ = 0;

    int timing_level_ = 0;
    int mem_level_ = 0;
    std::atomic<int64_t> timing_pre_us_{0};
    std::atomic<int64_t> timing_inf_us_{0};
    std::atomic<int64_t> timing_post_us_{0};
    std::atomic<int64_t> timing_pre_count_{0};
    std::atomic<int64_t> timing_inf_count_{0};
    std::atomic<int64_t> timing_post_count_{0};

    // Serial fallback (debug)
    std::unique_ptr<OverlapAddStreamer> serial_;

    // Pipelined state
    std::unique_ptr<OverlapAddStreamer> oa_;
    std::unique_ptr<ThreadSafeQueue<Task>> q_pre_;
    std::unique_ptr<ThreadSafeQueue<Task>> q_inf_;
    std::unique_ptr<ThreadSafeQueue<Task>> q_post_;
    std::unique_ptr<ThreadSafeQueue<Task>> q_done_;
    std::unique_ptr<ThreadSafeQueue<std::shared_ptr<ChunkState>>> q_free_state_;

    std::thread t_pre_;
    std::thread t_inf_;
    std::thread t_post_;

    size_t in_flight_ = 0;
    bool finalized_ = false;

    std::mutex exception_mutex_;
    std::exception_ptr exception_ = nullptr;

    bool joined_ = false;

    void PrintTimingSummaryIfEnabled() {
        if (timing_level_ <= 0) return;

        const int64_t pre_count = timing_pre_count_.load();
        const int64_t inf_count = timing_inf_count_.load();
        const int64_t post_count = timing_post_count_.load();
        if (pre_count == 0 && inf_count == 0 && post_count == 0) return;

        const double pre_ms = (pre_count > 0) ? (static_cast<double>(timing_pre_us_.load()) / 1000.0 / static_cast<double>(pre_count)) : 0.0;
        const double inf_ms = (inf_count > 0) ? (static_cast<double>(timing_inf_us_.load()) / 1000.0 / static_cast<double>(inf_count)) : 0.0;
        const double post_ms = (post_count > 0) ? (static_cast<double>(timing_post_us_.load()) / 1000.0 / static_cast<double>(post_count)) : 0.0;

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << "[StreamTiming] summary"
            << " chunks=" << std::max({pre_count, inf_count, post_count})
            << " pre_avg_ms=" << pre_ms
            << " inf_avg_ms=" << inf_ms
            << " post_avg_ms=" << post_ms;
        LogStreamTimingLine(oss.str());
    }

    void SetException(std::exception_ptr eptr) {
        {
            std::lock_guard<std::mutex> lock(exception_mutex_);
            if (!exception_) {
                exception_ = eptr;
            }
        }

        if (q_pre_) q_pre_->Shutdown();
        if (q_inf_) q_inf_->Shutdown();
        if (q_post_) q_post_->Shutdown();
        if (q_done_) q_done_->Shutdown();
        if (q_free_state_) q_free_state_->Shutdown();
    }

    void MaybeRethrow() {
        std::exception_ptr eptr;
        {
            std::lock_guard<std::mutex> lock(exception_mutex_);
            eptr = exception_;
        }
        if (eptr) {
            std::rethrow_exception(eptr);
        }
    }

    void ResetOut(std::vector<std::vector<float>>& out_acc) {
        if (num_stems_ <= 0) {
            out_acc.clear();
            return;
        }

        if (out_acc.size() != static_cast<size_t>(num_stems_)) {
            out_acc.resize(static_cast<size_t>(num_stems_));
        }

        const size_t reserve_frames =
            (step_ > 0) ? (static_cast<size_t>(step_) * pipeline_depth_) : static_cast<size_t>(std::max(0, chunk_size_));
        const size_t reserve_floats = reserve_frames * 2;

        for (int s = 0; s < num_stems_; ++s) {
            auto& v = out_acc[static_cast<size_t>(s)];
            v.clear();
            if (reserve_floats > 0 && v.capacity() < reserve_floats) {
                v.reserve(reserve_floats);
            }
        }
    }

    void MaybeLogMem(const char* tag, const Task& done) {
        if (mem_level_ <= 0) return;

        const ProcessMemInfo mem = GetProcessMemInfo();
        const double ws_mb = static_cast<double>(mem.working_set_bytes) / (1024.0 * 1024.0);
        const double priv_mb = static_cast<double>(mem.private_bytes) / (1024.0 * 1024.0);

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << "[StreamMem] " << tag
            << " offset=" << done.scheduled.offset
            << " part=" << done.scheduled.part_len
            << " in_flight=" << in_flight_
            << " ws_mb=" << ws_mb;

        if (mem.private_bytes > 0) {
            oss << " priv_mb=" << priv_mb;
        }

#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
            ggml_backend_t backend = engine_ && engine_->model_ ? engine_->model_->GetBackend() : nullptr;
            if (backend && ggml_backend_is_cuda(backend)) {
                size_t free_bytes = 0;
                size_t total_bytes = 0;
                cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
                if (err == cudaSuccess && total_bytes > 0) {
                    const double gpu_free_mb = static_cast<double>(free_bytes) / (1024.0 * 1024.0);
                    const double gpu_used_mb = static_cast<double>(total_bytes - free_bytes) / (1024.0 * 1024.0);
                    oss << " gpu_used_mb=" << gpu_used_mb
                        << " gpu_free_mb=" << gpu_free_mb;
                } else {
                    // clear the error
                    (void) cudaGetLastError();
                }
            }
        }
#endif

        if (mem_level_ >= 2 && done.state) {
            const auto& st = *done.state;
            oss << " n_frames=" << st.n_frames
                << " in_cap=" << st.input_audio.capacity()
                << " stft_cap=" << st.stft_flattened.capacity()
                << " mask_cap=" << st.mask_output.capacity();
            if (!st.final_audio.empty()) {
                oss << " out0_cap=" << st.final_audio[0].capacity();
            }
        }

        LogStreamTimingLine(oss.str());
    }

    void RecycleState(std::shared_ptr<ChunkState>&& st) {
        if (!st) return;
        if (!q_free_state_) return;
        q_free_state_->Push(std::move(st));
    }

    void DrainDone(std::vector<std::vector<float>>& out_acc) {
        Task done;
        while (q_done_ && q_done_->TryPop(done)) {
            if (in_flight_ == 0) {
                throw std::runtime_error("Internal error: in-flight underflow in streaming drain");
            }
            --in_flight_;

            oa_->ConsumeScheduledAppend(done.scheduled, done.state->final_audio, out_acc);
            MaybeLogMem("drain", done);
            RecycleState(std::move(done.state));
        }
    }

    void ScheduleMore() {
        if (!oa_) return;
        while (in_flight_ < pipeline_depth_) {
            OverlapAddStreamer::ScheduledChunk scheduled;
            if (!oa_->TryScheduleNextMeta(scheduled)) break;

            Task task;
            task.scheduled = std::move(scheduled);

            // Pre-allocate/reuse ChunkState for this chunk so we can reuse its input_audio buffer (avoids per-chunk heap growth).
            std::shared_ptr<ChunkState> state;
            if (q_free_state_) {
                if (!q_free_state_->Pop(state)) {
                    break;
                }
            }
            if (!state) {
                state = std::make_shared<ChunkState>();
            }

            state->id = task.scheduled.offset;
            oa_->MaterializeChunkInput(task.scheduled, state->input_audio);
            task.state = std::move(state);

            q_pre_->Push(std::move(task));
            ++in_flight_;
        }
    }

    void StartThreads() {
        t_pre_ = std::thread([this]() {
            try {
                Task task;
                while (q_pre_->Pop(task)) {
                    using clock = std::chrono::high_resolution_clock;
                    const auto t0 = (timing_level_ > 0) ? clock::now() : clock::time_point{};

                    if (!task.state) {
                        throw std::runtime_error("Internal error: missing ChunkState in streaming pre stage");
                    }

                    engine_->PreProcessChunkInPlace(*task.state);

                    if (timing_level_ > 0) {
                        const auto t1 = clock::now();
                        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                        timing_pre_us_.fetch_add(us);
                        timing_pre_count_.fetch_add(1);

                        if (timing_level_ >= 1) {
                            std::ostringstream oss;
                            oss << std::fixed << std::setprecision(2);
                            oss << "[StreamTiming] pre  offset=" << task.scheduled.offset
                                << " part=" << task.scheduled.part_len
                                << " n_frames=" << task.state->n_frames
                                << " ms=" << (static_cast<double>(us) / 1000.0);
                            LogStreamTimingLine(oss.str());
                        }
                    }
                    q_inf_->Push(std::move(task));
                }
            } catch (...) {
                SetException(std::current_exception());
            }
            q_inf_->Shutdown();
        });

        t_inf_ = std::thread([this]() {
            try {
                Task task;
                while (q_inf_->Pop(task)) {
                    using clock = std::chrono::high_resolution_clock;
                    const auto t0 = (timing_level_ > 0) ? clock::now() : clock::time_point{};

                    engine_->RunInference(task.state);

                    if (timing_level_ > 0) {
                        const auto t1 = clock::now();
                        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                        timing_inf_us_.fetch_add(us);
                        timing_inf_count_.fetch_add(1);

                        if (timing_level_ >= 1) {
                            std::ostringstream oss;
                            oss << std::fixed << std::setprecision(2);
                            oss << "[StreamTiming] inf  offset=" << task.scheduled.offset
                                << " part=" << task.scheduled.part_len
                                << " n_frames=" << task.state->n_frames
                                << " ms=" << (static_cast<double>(us) / 1000.0);
                            LogStreamTimingLine(oss.str());
                        }
                    }
                    q_post_->Push(std::move(task));
                }
            } catch (...) {
                SetException(std::current_exception());
            }
            q_post_->Shutdown();
        });

        t_post_ = std::thread([this]() {
            try {
                Task task;
                while (q_post_->Pop(task)) {
                    using clock = std::chrono::high_resolution_clock;
                    const auto t0 = (timing_level_ > 0) ? clock::now() : clock::time_point{};

                    engine_->PostProcessChunk(task.state);

                    if (timing_level_ > 0) {
                        const auto t1 = clock::now();
                        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                        timing_post_us_.fetch_add(us);
                        timing_post_count_.fetch_add(1);

                        if (timing_level_ >= 1) {
                            std::ostringstream oss;
                            oss << std::fixed << std::setprecision(2);
                            oss << "[StreamTiming] post offset=" << task.scheduled.offset
                                << " part=" << task.scheduled.part_len
                                << " n_frames=" << task.state->n_frames
                                << " ms=" << (static_cast<double>(us) / 1000.0);
                            LogStreamTimingLine(oss.str());
                        }
                    }
                    q_done_->Push(std::move(task));
                }
            } catch (...) {
                SetException(std::current_exception());
            }
            q_done_->Shutdown();
        });
    }

    void ShutdownAndJoin() {
        if (!pipelined_) return;
        if (joined_) return;
        joined_ = true;

        if (q_pre_) q_pre_->Shutdown();
        if (q_inf_) q_inf_->Shutdown();
        if (q_post_) q_post_->Shutdown();
        if (q_done_) q_done_->Shutdown();
        if (q_free_state_) q_free_state_->Shutdown();

        if (t_pre_.joinable()) t_pre_.join();
        if (t_inf_.joinable()) t_inf_.join();
        if (t_post_.joinable()) t_post_.join();
    }
};

Inference::StreamContext::StreamContext() = default;
Inference::StreamContext::~StreamContext() = default;

std::unique_ptr<Inference::StreamContext> Inference::CreateStream(int chunk_size, int num_overlap, bool pipelined) {
    int resolved_chunk = (chunk_size > 0) ? chunk_size : GetDefaultChunkSize();
    int resolved_overlap = (num_overlap > 0) ? num_overlap : GetDefaultNumOverlap();

    auto ctx = std::make_unique<StreamContext>();
    ctx->impl = std::make_unique<StreamImpl>(this, resolved_chunk, resolved_overlap, GetNumStems(), pipelined);
    return ctx;
}

std::vector<std::vector<float>> Inference::ProcessStream(StreamContext& ctx, const std::vector<float>& input_chunk) {
    std::vector<std::vector<float>> out_acc;
    ProcessStreamInto(ctx, input_chunk, out_acc);
    return out_acc;
}

void Inference::ProcessStreamInto(StreamContext& ctx,
                                  const std::vector<float>& input_chunk,
                                  std::vector<std::vector<float>>& out_acc) {
    if (ctx.finalized) {
        throw std::runtime_error("Error: stream already finalized");
    }
    if (!ctx.impl) {
        throw std::runtime_error("Error: invalid stream context");
    }
    ctx.impl->ProcessInto(input_chunk, out_acc);
}

std::vector<std::vector<float>> Inference::FinalizeStream(StreamContext& ctx) {
    std::vector<std::vector<float>> out_acc;
    FinalizeStreamInto(ctx, out_acc);
    return out_acc;
}

void Inference::FinalizeStreamInto(StreamContext& ctx, std::vector<std::vector<float>>& out_acc) {
    if (ctx.finalized) {
        throw std::runtime_error("Error: stream already finalized");
    }
    if (!ctx.impl) {
        throw std::runtime_error("Error: invalid stream context");
    }
    ctx.finalized = true;
    ctx.impl->FinalizeInto(out_acc);
}
