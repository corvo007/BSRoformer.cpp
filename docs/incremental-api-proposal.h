// Proposed Streaming API for Inference class
// This is a design proposal - not yet implemented

#pragma once
#include <vector>
#include <memory>

// Opaque handle for streaming session state
class StreamContext {
public:
    ~StreamContext();

private:
    friend class Inference;
    StreamContext(int chunk_size, int num_overlap, int num_stems);

    // Overlap-add accumulator
    std::vector<std::vector<float>> accumulator_;  // [stems][samples]
    std::vector<float> counter_;                   // Windowing counter

    // Input buffering
    std::vector<float> input_buffer_;              // Incomplete chunk
    int buffer_fill_ = 0;

    // Position tracking
    int write_position_ = 0;                       // Write offset in accumulator
    int output_position_ = 0;                      // Already returned to user

    // Config
    int chunk_size_;
    int num_overlap_;
    int step_;                                     // chunk_size / num_overlap
    int border_;                                   // chunk_size - step
    int fade_size_;                                // chunk_size / 10

    // State flags
    bool first_chunk_ = true;
    bool finalized_ = false;
};

// Add to Inference class:
class Inference {
public:
    // === New Streaming API ===

    // Initialize streaming session
    // Returns: Opaque context handle (user must keep alive)
    std::unique_ptr<StreamContext> CreateStream(
        int chunk_size = 352800,  // Use model defaults if not specified
        int num_overlap = 2
    );

    // Feed audio incrementally (interleaved stereo float32)
    // Returns: Stems ready for output [stems][samples]
    //          Empty vector if not enough data accumulated yet
    // Throws: std::runtime_error if context finalized or invalid input
    std::vector<std::vector<float>> ProcessStream(
        StreamContext& ctx,
        const std::vector<float>& input_chunk
    );

    // Flush remaining audio and get final output
    // Returns: Final stems [stems][samples]
    // Throws: std::runtime_error if already finalized
    // Note: Context becomes unusable after this call
    std::vector<std::vector<float>> FinalizeStream(
        StreamContext& ctx
    );

private:
    // Helper: Accumulate chunk output with windowing
    void AccumulateChunkOutput(
        StreamContext& ctx,
        const std::vector<std::vector<float>>& chunk_output,
        bool is_first,
        bool is_last
    );

    // Helper: Extract ready samples from accumulator
    std::vector<std::vector<float>> ExtractOutput(
        StreamContext& ctx,
        int num_samples
    );
};
