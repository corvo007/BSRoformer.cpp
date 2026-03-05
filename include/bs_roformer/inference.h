#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <cstdint>
#include <stdexcept>
// Forward declaration
class BSRoformer;

// Forward declaration
namespace ggml { struct context; struct cgraph; }

class Inference {
public:
    using CancelCallback = std::function<bool()>;

    Inference(const std::string& model_path);
    ~Inference();

    // Process a full audio track (interleaved stereo float32)
    // Uses overlap-add chunking to handle long files
    // Process a full audio track (interleaved stereo float32)
    // Returns a vector of stems, where each stem is an interleaved stereo float vector
    std::vector<std::vector<float>> Process(const std::vector<float>& input_audio, 
                               int chunk_size = 352800, 
                               int num_overlap = 2,
                               std::function<void(float)> progress_callback = nullptr,
                               CancelCallback cancel_callback = nullptr);

    // Low-level chunk processing (public for testing)
    std::vector<std::vector<float>> ProcessChunk(const std::vector<float>& chunk_audio);

    // Get model's recommended inference defaults
    int GetDefaultChunkSize() const;
    int GetDefaultNumOverlap() const;
    int GetSampleRate() const;
    int GetNumStems() const;

    // Static helper for Overlap-Add logic (matches Python exactly)
    // model_func: input [samples], output [stems][samples] (interleaved stereo)
    using ModelCallback = std::function<std::vector<std::vector<float>>(const std::vector<float>&)>;
    static std::vector<std::vector<float>> ProcessOverlapAdd(const std::vector<float>& input_audio, 
                                                 int chunk_size, 
                                                 int num_overlap,
                                                 ModelCallback model_func,
                                                 std::function<void(float)> progress_callback = nullptr,
                                                 CancelCallback cancel_callback = nullptr);

    // =============================================================================================
    // Streaming API (constant-memory overlap-add)
    // =============================================================================================

    class OverlapAddStreamer {
    public:
        OverlapAddStreamer(int chunk_size, int num_overlap, int num_stems, ModelCallback model_func);

        struct ScheduledChunk {
            int64_t offset = 0;            // padded-domain offset (samples-per-channel)
            int part_len = 0;              // valid samples in chunk_out to accumulate
            bool is_first = false;
            bool is_last = false;
            std::vector<float> chunk_in;   // interleaved stereo, length = chunk_size*2
        };

        // Manual/pipelined API:
        // - Feed(): append input samples (may buffer until padding decision is possible)
        // - FinalizeInput(): signal end-of-input (materializes right pad if needed)
        // - TryScheduleNext(): extract next chunk for processing (FIFO order)
        // - ConsumeScheduled(): consume processed output for a scheduled chunk (must be in order)
        void Feed(const std::vector<float>& input_audio);
        void FinalizeInput();
        bool TryScheduleNext(ScheduledChunk& out);

        // Like TryScheduleNext(), but does NOT allocate/copy chunk_in.
        // Call MaterializeChunkInput() to fill a reusable buffer when needed.
        bool TryScheduleNextMeta(ScheduledChunk& out);

        // Materialize scheduled.chunk_in into dst (interleaved stereo, length = chunk_size*2).
        // This lets pipelined inference reuse buffers (constant-memory, less heap churn).
        void MaterializeChunkInput(const ScheduledChunk& scheduled, std::vector<float>& dst) const;
        std::vector<std::vector<float>> ConsumeScheduled(const ScheduledChunk& scheduled,
                                                         const std::vector<std::vector<float>>& chunk_out);

        // Like ConsumeScheduled(), but appends any ready output into out_acc (reusing buffers to avoid per-chunk alloc).
        void ConsumeScheduledAppend(const ScheduledChunk& scheduled,
                                    const std::vector<std::vector<float>>& chunk_out,
                                    std::vector<std::vector<float>>& out_acc);

        // Feed interleaved stereo samples (size must be even). Returns ready output (may be empty).
        std::vector<std::vector<float>> Push(const std::vector<float>& input_audio);

        // Finish stream and return remaining output. After calling this, Push() throws.
        std::vector<std::vector<float>> Finalize();

    private:
        static constexpr int kChannels = 2;

        int chunk_size_ = 0;
        int num_overlap_ = 0;
        int step_ = 0;
        int border_ = 0;
        int fade_size_ = 0;
        int num_stems_ = 0;
        ModelCallback model_func_ = nullptr;

        bool finalized_ = false;
        bool decided_pad_ = false;
        bool do_pad_ = false;

        // Original (non-padded) input length, in samples-per-channel
        int64_t total_input_samples_ = 0;

        // For right reflection padding (stores tail of original input, interleaved stereo)
        std::vector<float> tail_buffer_;

        // Buffer before we can decide do_pad (interleaved stereo, original domain)
        std::vector<float> prebuffer_;

        // Main padded-domain input buffer (interleaved stereo). Logical start index tracked separately.
        std::vector<float> input_buffer_;
        int64_t input_base_ = 0;      // absolute padded sample index (per-channel) of input_buffer_[input_offset_]
        size_t input_offset_ = 0;     // offset into input_buffer_ in floats

        // Sliding overlap-add buffers (padded domain, window length = chunk_size_)
        std::vector<std::vector<float>> accum_; // [stems][chunk_size_*2]
        std::vector<float> counter_;            // [chunk_size_] (per-sample, same for both channels)
        std::vector<float> window_base_;        // [chunk_size_]

        // Current processing position (padded domain), in samples-per-channel
        int64_t current_offset_ = 0;

        // Cropping info
        int64_t crop_left_ = 0;                 // pad_l (0 or border_)
        int64_t produced_padded_samples_ = 0;   // padded samples emitted/advanced (per-channel)
        int64_t emitted_output_samples_ = 0;    // output samples emitted after cropping (per-channel)

        // Manual scheduling state
        bool input_finalized_ = false;
        int64_t total_length_padded_ = 0;
        int64_t schedule_offset_ = 0;

        void DecidePaddingIfPossible(bool flushing);
        void MaterializeLeftPadAndMovePrebuffer();
        void AppendToInputBuffer(const std::vector<float>& interleaved);
        int64_t AvailablePaddedSamples() const;
        void ReadFromInputBuffer(int64_t abs_sample_idx, int part_len, std::vector<float>& dst) const;
        void DropInputPrefixUpTo(int64_t new_base_sample);
        std::vector<float> BuildRightPad() const;

        struct ChunkExtractResult {
            std::vector<float> chunk_in; // size chunk_size_*2
            int part_len = 0;
        };
        ChunkExtractResult ExtractChunkAtOffset(int64_t abs_offset, int64_t total_length_padded, bool flushing) const;

        void AccumulateChunk(const std::vector<std::vector<float>>& chunk_out, int part_len, bool is_first, bool is_last);
        void AppendReadyOutput(int ready_samples, int64_t remaining_output_limit /*<0 means no limit*/, std::vector<std::vector<float>>& out_acc);
        std::vector<std::vector<float>> EmitReadyOutput(int ready_samples, int64_t remaining_output_limit /*<0 means no limit*/);
        void ShiftWindowByStep();
    };

    struct StreamImpl;

    struct StreamContext {
        StreamContext();
        ~StreamContext();
        StreamContext(const StreamContext&) = delete;
        StreamContext& operator=(const StreamContext&) = delete;

        std::unique_ptr<StreamImpl> impl;
        bool finalized = false;
    };

    // Create a streaming session. Passing -1 uses model defaults.
    std::unique_ptr<StreamContext> CreateStream(int chunk_size = -1, int num_overlap = -1, bool pipelined = true);

    // Feed interleaved stereo samples; returns ready output (may be empty).
    std::vector<std::vector<float>> ProcessStream(StreamContext& ctx, const std::vector<float>& input_chunk);

    // Like ProcessStream(), but writes output into out_acc, reusing its capacity to reduce heap growth.
    void ProcessStreamInto(StreamContext& ctx, const std::vector<float>& input_chunk, std::vector<std::vector<float>>& out_acc);

    // Flush remaining audio; ctx becomes finalized.
    std::vector<std::vector<float>> FinalizeStream(StreamContext& ctx);

    // Like FinalizeStream(), but writes output into out_acc, reusing its capacity to reduce heap growth.
    void FinalizeStreamInto(StreamContext& ctx, std::vector<std::vector<float>>& out_acc);

  private:
    // Pipelined Overlap-Add
    std::vector<std::vector<float>> ProcessOverlapAddPipelined(const std::vector<float>& input_audio, 
                                                   int chunk_size, 
                                                  int num_overlap,
                                                  std::function<void(float)> progress_callback,
                                                  CancelCallback cancel_callback);

private:
    std::unique_ptr<BSRoformer> model_;
    
    // Persistent Graph State
    struct ggml_context* ctx_ = nullptr;
    struct ggml_cgraph* gf_ = nullptr;
    struct ggml_gallocr* allocr_ = nullptr;
    std::vector<uint8_t> ctx_mem_; // backing store for ggml context (avoids ggml internal malloc/assert)
     
    // Cached Input Tensors (owned by ctx_)
    struct ggml_tensor* input_tensor_ = nullptr;
    struct ggml_tensor* pos_time_ = nullptr;
    struct ggml_tensor* pos_freq_ = nullptr;
    struct ggml_tensor* mask_out_tensor_ = nullptr;

    // Optional pinned staging buffers for CUDA H2D/D2H copies.
    // Keeps memory usage stable when using async copies from pageable memory.
    void* cuda_pinned_in_ = nullptr;
    size_t cuda_pinned_in_bytes_ = 0;
    void* cuda_pinned_out_ = nullptr;
    size_t cuda_pinned_out_bytes_ = 0;
    void* cuda_pinned_pos_time_ = nullptr;
    size_t cuda_pinned_pos_time_bytes_ = 0;
    void* cuda_pinned_pos_freq_ = nullptr;
    size_t cuda_pinned_pos_freq_bytes_ = 0;

    // Cached Host Data (to avoid reallocation)
    std::vector<int32_t> pos_time_data_;
    std::vector<int32_t> pos_freq_data_;
    int uploaded_pos_n_frames_ = -1;

    // Cached window (win_length) for STFT/ISTFT (read-only after init)
    std::vector<float> hann_window_;

    // Current config state
    int cached_n_frames_ = -1;

    // Pipelined State Data
    struct ChunkState {
        int64_t id = -1;
        std::vector<float> input_audio;       // Original chunk audio
        std::vector<float> stft_flattened;    // [Prepared Input for GPU]
        std::vector<std::vector<float>> stft_outputs; // Kept for reconstruction
        int n_frames = 0;
        
        std::vector<float> mask_output;       // Output from GPU
        std::vector<std::vector<float>> final_audio;       // Result after ISTFT [stems][samples]

        // Optional CUDA host registration state for large per-chunk I/O buffers.
        // Implemented in src/inference.cpp (no CUDA headers required here).
        void* cuda_reg_stft_ptr = nullptr;
        size_t cuda_reg_stft_bytes = 0;
        void* cuda_reg_mask_ptr = nullptr;
        size_t cuda_reg_mask_bytes = 0;

        ~ChunkState();
    };

    // Helper to ensure graph is built for specific n_frames
    bool EnsureGraph(int n_frames);

    void ComputeSTFT(const std::vector<float>& input_audio,
                     std::vector<std::vector<float>>& stft_outputs,
                     int& n_frames);
                     
    void PrepareModelInput(const std::vector<std::vector<float>>& stft_outputs,
                           int n_frames,
                           std::vector<float>& model_input_rearranged);

    void PostProcessAndISTFT(const std::vector<float>& mask_output,
                             const std::vector<std::vector<float>>& stft_outputs,
                             int n_frames,
                             std::vector<std::vector<float>>& output_audio);

    // Pipeline Steps
    void PreProcessChunkInto(ChunkState& state, std::vector<float> chunk_audio, int64_t id);
    void PreProcessChunkInPlace(ChunkState& state);
    std::shared_ptr<ChunkState> PreProcessChunk(std::vector<float> chunk_audio, int64_t id);
    void RunInference(std::shared_ptr<ChunkState> state);
    void PostProcessChunk(std::shared_ptr<ChunkState> state);
};
