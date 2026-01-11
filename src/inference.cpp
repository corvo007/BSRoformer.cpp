#include "mel_band_roformer/inference.h"
#include "model.h"
#include "utils.h"
#include "stft.h"
#include <iostream>
#include <complex>
#include <algorithm>
#include <cstring>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <chrono>
#include <future>

using Complex = std::complex<float>;

// Helper forward decl
std::vector<float> GetWindow(int size, int fade_size);

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


Inference::Inference(const std::string& model_path) {
    model_ = std::make_unique<MelBandRoformer>();
    model_->Initialize(model_path);
}

int Inference::GetDefaultChunkSize() const {
    return model_->GetDefaultChunkSize();
}

int Inference::GetDefaultNumOverlap() const {
    return model_->GetDefaultNumOverlap();
}

Inference::~Inference() {
    if (allocr_) ggml_gallocr_free(allocr_);
    if (ctx_) ggml_free(ctx_);
    // gf_ is part of ctx_, tensor pointers are part of ctx_
}

bool Inference::EnsureGraph(int n_frames) {
    if (n_frames == cached_n_frames_ && ctx_ != nullptr) {
        return true;
    }
    
    std::cout << "[Inference] Building graph for n_frames=" << n_frames << std::endl;

    // Cleanup old graph
    if (allocr_) { ggml_gallocr_free(allocr_); allocr_ = nullptr; }
    if (ctx_) { ggml_free(ctx_); ctx_ = nullptr; }
    
    cached_n_frames_ = n_frames;

    // Allocate context
    size_t mem_size = 1024ull * 1024 * 1024; // 1GB
    struct ggml_init_params ctx_params = { mem_size, nullptr, true };
    ctx_ = ggml_init(ctx_params);
    if (!ctx_) return false;
    
    gf_ = ggml_new_graph_custom(ctx_, 65536, false);

    int batch = 1;
    int total_dim_input = model_->GetTotalDimInput();
    
    input_tensor_ = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, total_dim_input, n_frames, batch);
    ggml_set_input(input_tensor_);

    // BandSplit -> Transformers -> MaskEstimator
    ggml_tensor* band_out = model_->BuildBandSplitGraph(ctx_, input_tensor_, gf_, n_frames, batch);
    
    int n_bands = model_->GetNumBands();
    pos_time_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_frames * n_bands);
    pos_freq_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_bands * n_frames);
    ggml_set_input(pos_time_);
    ggml_set_input(pos_freq_);

    ggml_tensor* trans_out = model_->BuildTransformersGraph(ctx_, band_out, gf_, pos_time_, pos_freq_, n_frames, batch);
    mask_out_tensor_ = model_->BuildMaskEstimatorGraph(ctx_, trans_out, gf_, n_frames, batch);

    // Allocate compute buffer (VRAM)
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model_->GetBackend()));
    if (!ggml_gallocr_alloc_graph(allocr_, gf_)) {
        std::cerr << "[Inference] Failed to allocate graph VRAM" << std::endl;
        return false;
    }
    
    return true;
}

void Inference::ComputeSTFT(const std::vector<float>& input_audio,
                            std::vector<std::vector<float>>& stft_outputs,
                            int& n_frames) {
    int n_fft = model_->GetNFFT();
    int hop_length = model_->GetHopLength();
    int win_length = model_->GetWinLength();
    int n_freq = n_fft / 2 + 1;
    int channels = 2; 

    std::vector<float> window(win_length);
    stft::hann_window(window.data(), win_length);

    stft_outputs.resize(channels);
    int n_samples = input_audio.size() / channels;

    for (int ch = 0; ch < channels; ++ch) {
        std::vector<float> channel_audio(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            channel_audio[i] = input_audio[ch + i * channels];
        }

        stft_outputs[ch].resize(n_freq * (n_samples / hop_length + 5) * 2);
        stft::compute_stft(channel_audio.data(), n_samples, n_fft, hop_length, win_length, 
                           window.data(), true, stft_outputs[ch].data(), &n_frames);
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

    std::vector<float> window(win_length);
    stft::hann_window(window.data(), win_length);
    
    // Process each stem
    for (int stem = 0; stem < num_stems; ++stem) {
        std::vector<Complex> masks(num_freq_indices * n_frames);
        
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
                masks[f * n_frames + t] = Complex(mask_output[idx + 0], mask_output[idx + 1]);
            }
        }

        int total_freq_stereo = n_freq * channels;
        std::vector<Complex> masks_summed(total_freq_stereo * n_frames, {0.0f, 0.0f});

        for (int f = 0; f < num_freq_indices; ++f) {
            int dst_idx_base = freq_indices[f]; 
            for (int t = 0; t < n_frames; ++t) {
                masks_summed[dst_idx_base * n_frames + t] += masks[f * n_frames + t];
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
                    masks_summed[freq_stereo_idx * n_frames + t] /= denom;
                }
            }
        }

        std::vector<Complex> stft_output_masked(total_freq_stereo * n_frames);
        #ifdef USE_OPENMP
        #pragma omp parallel for
        #endif
        for (int ch = 0; ch < channels; ++ch) {
            for (int f = 0; f < n_freq; ++f) {
                int freq_stereo_idx = f * channels + ch;
                for (int t = 0; t < n_frames; ++t) {
                    int mask_idx = freq_stereo_idx * n_frames + t;
                    int stft_idx = (f * n_frames + t) * 2;
                    Complex stft_val(stft_outputs[ch][stft_idx + 0], stft_outputs[ch][stft_idx + 1]);
                    stft_output_masked[mask_idx] = stft_val * masks_summed[mask_idx];
                }
            }
        }

        std::vector<std::vector<float>> output_channels(channels);
        int n_samples_out = 0;

        for (int ch = 0; ch < channels; ++ch) {
            std::vector<float> istft_in(n_freq * n_frames * 2);
            for (int f = 0; f < n_freq; ++f) {
                int freq_stereo_idx = f * channels + ch;
                for (int t = 0; t < n_frames; ++t) {
                    int mask_idx = freq_stereo_idx * n_frames + t;
                    int dst_idx = (f * n_frames + t) * 2;
                    istft_in[dst_idx + 0] = stft_output_masked[mask_idx].real();
                    istft_in[dst_idx + 1] = stft_output_masked[mask_idx].imag();
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
            output_channels[ch].resize(approx_len + n_fft); 
            stft::compute_istft(istft_in.data(), n_freq, n_frames, n_fft, hop_length, win_length, 
                                window.data(), true, approx_len, output_channels[ch].data());
            if (ch == 0) n_samples_out = approx_len;
            output_channels[ch].resize(n_samples_out);
        }

        output_audio[stem].resize(channels * n_samples_out);
        for (int i = 0; i < n_samples_out; ++i) {
            for (int ch = 0; ch < channels; ++ch) {
                output_audio[stem][ch + i * channels] = output_channels[ch][i];
            }
        }
    }
}

#include <future>

std::vector<std::vector<float>> Inference::Process(const std::vector<float>& input_audio, int chunk_size, int num_overlap, std::function<void(float)> progress_callback) {
    if (input_audio.empty()) return {};
    return ProcessOverlapAddPipelined(input_audio, chunk_size, num_overlap, progress_callback);
}

// =================================================================================================
// Pipeline Stages
// =================================================================================================

std::shared_ptr<Inference::ChunkState> Inference::PreProcessChunk(const std::vector<float>& chunk_audio, int id) {
    auto state = std::make_shared<ChunkState>();
    state->id = id;
    state->input_audio = chunk_audio; // Copy

    if (chunk_audio.empty()) return state;

    // 1. STFT
    ComputeSTFT(state->input_audio, state->stft_outputs, state->n_frames);

    // 2. Prepare Input
    PrepareModelInput(state->stft_outputs, state->n_frames, state->stft_flattened);

    return state;
}

void Inference::RunInference(std::shared_ptr<ChunkState> state) {
    if (!state || state->stft_flattened.empty()) return;

    // 3. Ensure Graph
    if (!EnsureGraph(state->n_frames)) {
        return;
    }

    int n_bands = model_->GetNumBands();
    int n_frames = state->n_frames;

    // Prepare position data
    // TODO: Cache these to avoid allocation every frame if size is constant
    std::vector<int32_t> pos_time_data(n_frames * n_bands);
    for(int i=0; i < n_frames * n_bands; ++i) pos_time_data[i] = i % n_frames;
    
    std::vector<int32_t> pos_freq_data(n_bands * n_frames);
    for(int i=0; i < n_bands * n_frames; ++i) pos_freq_data[i] = i % n_bands;

    // 4. Host -> Device
    ggml_backend_tensor_set(input_tensor_, state->stft_flattened.data(), 0, ggml_nbytes(input_tensor_));
    ggml_backend_tensor_set(pos_time_, pos_time_data.data(), 0, ggml_nbytes(pos_time_));
    ggml_backend_tensor_set(pos_freq_, pos_freq_data.data(), 0, ggml_nbytes(pos_freq_));

    // 5. Compute
    ggml_backend_graph_compute(model_->GetBackend(), gf_);

    // 6. Device -> Host
    state->mask_output.resize(ggml_nelements(mask_out_tensor_));
    ggml_backend_tensor_get(mask_out_tensor_, state->mask_output.data(), 0, ggml_nbytes(mask_out_tensor_));
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
}

std::vector<std::vector<float>> Inference::ProcessChunk(const std::vector<float>& chunk_audio) {
    // Serial fallback
    auto state = PreProcessChunk(chunk_audio, 0);
    RunInference(state);
    PostProcessChunk(state);
    return state->final_audio;
}

// =================================================================================================
// Pipelined Overlap-Add Logic
// =================================================================================================

std::vector<std::vector<float>> Inference::ProcessOverlapAddPipelined(const std::vector<float>& input_audio, 
                                                         int chunk_size, 
                                                         int num_overlap,
                                                         std::function<void(float)> progress_callback) {
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
    auto accumulate_result = [&](std::shared_ptr<ChunkState> state, int i) {
        if (!state) return;
        const std::vector<std::vector<float>>& chunk_out_stems = state->final_audio; // Now [stems][samples]
        if (chunk_out_stems.empty()) return;
        
        // Lazy Initialize result
        if (result.empty()) {
            int num_stems = chunk_out_stems.size();
            result.resize(num_stems, std::vector<float>(n_padded_samples * channels, 0.0f));
        }

        int remaining = n_padded_samples - i;
        int part_len = std::min(C, remaining); 

        std::vector<float> window = window_base; // Copy
        if (i == 0) {
            for(int k=0; k<fade_size; ++k) window[k] = 1.0f;
        } else if (i + step >= n_padded_samples) {
            for(int k=0; k<fade_size; ++k) window[C - 1 - k] = 1.0f;
        }
        
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
            
            // Counter is same for all stems, just update once
            counter[res_idx + 0] += w;
            counter[res_idx + 1] += w;
        }
    };

    // ==========================================================
    // Pipeline Loop
    // ==========================================================
    
    // Future for the NEXT chunk's preprocessing
    std::future<std::shared_ptr<ChunkState>> next_prep_future;
    
    // Future for the PREVIOUS chunk's postprocessing
    std::future<void> prev_post_future;
    
    std::shared_ptr<ChunkState> prev_state = nullptr;
    
    int i = 0;
    int current_offset = 0;
    
    // Bootstrap: Start PreProcessing first chunk
    {
        std::vector<float> chunk0 = extract_chunk(0);
        // Async launch
        next_prep_future = std::async(std::launch::async, 
            [this](std::vector<float> c, int id) { return this->PreProcessChunk(c, id); }, 
            std::move(chunk0), 0);
    }
    
    while (current_offset < n_padded_samples) {
        // 1. Wait for PRE-processing of CURRENT chunk
        if (next_prep_future.valid()) {
            // This blocks until STFT is done.
            // In steady state, this should be ready or nearly ready while GPU was busy.
        }
        auto current_state = next_prep_future.get();
        
        // 2. Start PRE-processing of NEXT chunk (if exists)
        int next_offset = current_offset + step;
        if (next_offset < n_padded_samples) {
             std::vector<float> chunk_next = extract_chunk(next_offset);
             next_prep_future = std::async(std::launch::async, 
                [this](std::vector<float> c, int id) { return this->PreProcessChunk(c, id); }, 
                std::move(chunk_next), next_offset);
        } else {
            // No more next chunks
        }
        
        // 3. Run Inference on CURRENT chunk (GPU Sync)
        // This blocks heavily.
        RunInference(current_state);
        
        // 4. Wait for POST-processing of PREVIOUS chunk
        if (prev_post_future.valid()) {
            prev_post_future.get();
        }
        
        // 5. Accumulate PREVIOUS chunk result (Serial, fast)
        // Note: PostProcessChunk fills 'final_audio', but doesn't accumulate to 'result'.
        // We do accumulation here on main thread to avoid races on 'result' buffer.
        if (prev_state) {
            int prev_offset = current_offset - step;
            accumulate_result(prev_state, prev_offset);
            prev_state = nullptr; // Free memory
        }
        
        // 6. Start POST-processing of CURRENT chunk
        prev_state = current_state;
        // Use shared_ptr copy
        prev_post_future = std::async(std::launch::async, 
            [this](std::shared_ptr<ChunkState> s) { this->PostProcessChunk(s); }, 
            prev_state);
            
        // Advance
        current_offset += step;

        if (progress_callback) {
            float progress = (float)std::min(current_offset, n_padded_samples) / n_padded_samples;
            progress_callback(progress);
        }
    }
    
    // Drain Pipeline
    // Wait for last post-process
    if (prev_post_future.valid()) {
        prev_post_future.get();
    }
    if (prev_state) {
        int prev_offset = current_offset - step;
        accumulate_result(prev_state, prev_offset);
    }
    
    // Normalize and Crop
    // result is [stems][samples]
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

std::vector<std::vector<float>> Inference::ProcessOverlapAdd(const std::vector<float>& input_audio, 
                                                int chunk_size, 
                                                int num_overlap,
                                                ModelCallback model_func,
                                                std::function<void(float)> progress_callback) {
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
