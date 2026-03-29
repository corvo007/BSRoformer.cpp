# Incremental Processing API Design

## Overview

This document proposes a new streaming API for `Inference` class that enables incremental audio processing without loading the entire audio file into memory.

## Current Architecture Analysis

### Existing API
```cpp
std::vector<std::vector<float>> Process(
    const std::vector<float>& input_audio,
    int chunk_size = 352800,
    int num_overlap = 2,
    std::function<void(float)> progress_callback = nullptr,
    CancelCallback cancel_callback = nullptr
);
```

**Limitations:**
- Requires complete audio in memory (`input_audio`)
- Allocates full-length buffers at line 515, 563 (`n_padded_samples * channels`)
- Cannot start processing until entire file is loaded

### Overlap-Add Mechanism
- **chunk_size**: Processing window (e.g., 352800 samples = 8 seconds @ 44.1kHz)
- **num_overlap**: Overlap factor (default 2)
- **step**: `chunk_size / num_overlap` (e.g., 176400 samples)
- **border**: `chunk_size - step` (overlap region)
- **fade_size**: `chunk_size / 10` (crossfade window)

**Key Insight:** Chunks are processed independently, then blended using windowed overlap-add.

### GGML Graph State
From `inference.h` lines 61-76:
- Graph is **cached** and reused (`ctx_`, `gf_`, `allocr_`)
- Graph is rebuilt only when `n_frames` changes (`EnsureGraph()`)
- No cross-chunk state in Transformer (stateless processing)

## Proposed API Design

### 1. Stream Context Handle

```cpp
class StreamContext {
public:
    // Opaque handle - implementation details hidden
    ~StreamContext();

private:
    friend class Inference;
    StreamContext(int chunk_size, int num_overlap, int num_stems);

    // Overlap-add state
    std::vector<std::vector<float>> accumulator_;  // [stems][samples]
    std::vector<float> counter_;                   // [samples]
    int write_position_ = 0;                       // Current write offset
    int output_position_ = 0;                      // Samples already returned

    // Buffering for overlap region
    std::vector<float> input_buffer_;              // Holds incomplete chunk
    int buffer_fill_ = 0;

    // Config
    int chunk_size_;
    int num_overlap_;
    int step_;
    int border_;
    int fade_size_;
    bool first_chunk_ = true;
    bool finalized_ = false;
};
```

### 2. New Public API

```cpp
class Inference {
public:
    // === Existing API (unchanged) ===
    std::vector<std::vector<float>> Process(...);

    // === New Streaming API ===

    // Initialize streaming session
    std::unique_ptr<StreamContext> CreateStream(
        int chunk_size = 352800,
        int num_overlap = 2
    );

    // Feed audio incrementally (interleaved stereo)
    // Returns stems ready for output [stems][samples]
    std::vector<std::vector<float>> ProcessStream(
        StreamContext& ctx,
        const std::vector<float>& input_chunk
    );

    // Flush remaining audio and get final output
    std::vector<std::vector<float>> FinalizeStream(
        StreamContext& ctx
    );
};
```

### 3. State Management Strategy

**Buffering Logic:**
- Input buffer accumulates samples until `chunk_size` is reached
- Process complete chunks using existing `ProcessChunk()`
- Maintain overlap region in accumulator for windowed blending

**Output Strategy:**
- Return samples beyond the overlap region (safe from future modifications)
- Keep `border` samples in accumulator for next chunk blending
- On finalize, return all remaining samples

**GGML Graph:**
- No changes needed - graph is already stateless and cached
- `EnsureGraph()` handles dynamic frame count

### 4. Usage Example

```cpp
// Initialize
Inference inference("model.gguf");
auto stream = inference.CreateStream(352800, 2);

// Process file incrementally (e.g., 1-second chunks)
const int READ_SIZE = 44100 * 2; // 1 sec stereo @ 44.1kHz
std::vector<float> read_buffer(READ_SIZE);

while (file.read(read_buffer)) {
    auto output = inference.ProcessStream(*stream, read_buffer);

    // Output is ready immediately - write to file/player
    for (int s = 0; s < output.size(); ++s) {
        write_stem(s, output[s]);
    }
}

// Flush remaining audio
auto final_output = inference.FinalizeStream(*stream);
for (int s = 0; s < final_output.size(); ++s) {
    write_stem(s, final_output[s]);
}
```

### 5. Implementation Details

**ProcessStream() Flow:**
1. Append `input_chunk` to `input_buffer_`
2. While `buffer_fill_ >= chunk_size_`:
   - Extract chunk from buffer
   - Call `ProcessChunk()` (existing method)
   - Accumulate with windowing
   - Shift buffer
3. Return samples from `output_position_` to `write_position_ - border_`
4. Update `output_position_`

**FinalizeStream() Flow:**
1. Pad remaining buffer to `chunk_size_` (reflect padding)
2. Process final chunk
3. Return all samples from `output_position_` to end
4. Mark context as finalized

### 6. Memory Comparison

**Current API:**
- Input: `N` samples
- Accumulator: `N + 2*border` samples × stems
- Peak: ~3× input size

**Streaming API:**
- Input buffer: `chunk_size` samples
- Accumulator: `chunk_size + border` samples × stems
- Peak: ~constant (independent of file length)

**Example:** 10-minute file @ 44.1kHz stereo
- Current: ~211 MB
- Streaming: ~13 MB (16× reduction)

### 7. Backward Compatibility

**Fully Compatible:**
- Existing `Process()` API unchanged
- New API is additive only
- No changes to `ProcessChunk()` signature
- GGML graph building logic untouched

**Migration Path:**
```cpp
// Old code continues to work
auto result = inference.Process(full_audio);

// New code for streaming
auto stream = inference.CreateStream();
while (has_data) {
    auto chunk_result = inference.ProcessStream(*stream, chunk);
}
auto final = inference.FinalizeStream(*stream);
```

### 8. Error Handling

```cpp
// Invalid state transitions
ProcessStream(finalized_context);  // throws std::runtime_error
FinalizeStream(finalized_context); // throws std::runtime_error

// Invalid input
ProcessStream(ctx, odd_length_audio); // throws (must be stereo)
```

### 9. Thread Safety

**Not Thread-Safe:**
- `StreamContext` is single-threaded (same as current API)
- Multiple streams can run in parallel (different contexts)
- GGML graph is shared but protected by existing mechanisms

### 10. API Header Changes

**include/bs_roformer/inference.h:**
```cpp
// Forward declaration
class StreamContext;

class Inference {
public:
    // ... existing methods ...

    // Streaming API
    std::unique_ptr<StreamContext> CreateStream(
        int chunk_size = 352800,
        int num_overlap = 2
    );

    std::vector<std::vector<float>> ProcessStream(
        StreamContext& ctx,
        const std::vector<float>& input_chunk
    );

    std::vector<std::vector<float>> FinalizeStream(
        StreamContext& ctx
    );

private:
    // Helper for stream processing
    void AccumulateChunkOutput(
        StreamContext& ctx,
        const std::vector<std::vector<float>>& chunk_output,
        bool is_first,
        bool is_last
    );
};
```

### 11. Implementation Complexity

**Low Risk:**
- Reuses existing `ProcessChunk()` - no model changes
- Overlap-add logic already proven in `ProcessOverlapAddPipelined()`
- No GGML graph modifications needed

**Estimated Effort:**
- Core implementation: ~200 lines
- Testing: ~100 lines
- Total: 1-2 days

### 12. Testing Strategy

```cpp
// Verify equivalence
std::vector<float> full_audio = load_audio("test.wav");

// Method 1: Batch
auto batch_result = inference.Process(full_audio);

// Method 2: Stream
auto stream = inference.CreateStream();
std::vector<std::vector<float>> stream_result;
for (size_t i = 0; i < full_audio.size(); i += CHUNK) {
    auto chunk = std::vector<float>(
        full_audio.begin() + i,
        full_audio.begin() + std::min(i + CHUNK, full_audio.size())
    );
    auto output = inference.ProcessStream(*stream, chunk);
    append(stream_result, output);
}
auto final = inference.FinalizeStream(*stream);
append(stream_result, final);

// Assert: batch_result ≈ stream_result (within numerical tolerance)
```

## Summary

**Proposed API:**
- `CreateStream()` - Initialize streaming session
- `ProcessStream()` - Feed audio incrementally, get output
- `FinalizeStream()` - Flush and complete

**Benefits:**
- Constant memory usage (independent of file length)
- Immediate output (low latency)
- Backward compatible
- Low implementation risk

**Trade-offs:**
- Slightly more complex API (3 calls vs 1)
- User must manage `StreamContext` lifetime
- No progress callback in streaming mode (user tracks input position)
