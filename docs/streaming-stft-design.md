# Streaming STFT/ISTFT Architecture Design

## Current Implementation Analysis

**Location**: `src/stft.h`, `src/inference.cpp:131-143`

**Current Approach**:
- Allocates full STFT buffer: `n_freq * (n_samples / hop_length + 5) * 2` floats
- Processes entire audio in one pass with OpenMP parallelization
- Memory: ~40MB for 3-minute stereo audio (n_fft=4096, hop=1024, 44.1kHz)

**Bottleneck**: Line 140 pre-allocates entire output buffer before processing.

---

## Streaming Architecture Design

### 1. Core Class Interface

```cpp
class StreamingSTFT {
public:
    StreamingSTFT(int n_fft, int hop_length, int win_length, bool center = true);

    // Feed audio incrementally, returns frames ready for processing
    int ProcessChunk(const float* audio, int n_samples, float* output, int max_frames);

    // Flush remaining buffered audio
    int Flush(float* output, int max_frames);

    // Reset state for new stream
    void Reset();

    int GetFramesReady() const { return frames_ready_; }

private:
    // Parameters
    int n_fft_, hop_length_, win_length_;
    bool center_;
    int n_freq_;  // n_fft / 2 + 1

    // Sliding window buffer
    std::vector<float> input_buffer_;
    int buffer_fill_;

    // Window coefficients (precomputed)
    std::vector<float> window_padded_;

    // FFT workspace
    stft::STFTBuffer fft_buffer_;

    // Frame counter
    int frames_ready_;
    bool flushed_;
};
```

---

### 2. Sliding Window Buffer Management

**Buffer Size**: `n_fft + (max_chunk_frames - 1) * hop_length`

**Processing Logic**:
```
Input Buffer:  [========|====|====|====]
               ^        ^    ^    ^
               |        |    |    |
            n_fft    hop  hop  hop

Frame 0: [n_fft samples starting at offset 0]
Frame 1: [n_fft samples starting at offset hop_length]
Frame 2: [n_fft samples starting at offset 2*hop_length]
...
```

**Algorithm**:
1. Append new audio to `input_buffer_[buffer_fill_:]`
2. While `buffer_fill_ >= n_fft_`:
   - Extract frame at offset `frames_ready_ * hop_length_`
   - Apply window, compute FFT, write to output
   - Increment `frames_ready_`
   - If `(frames_ready_ + 1) * hop_length_ + n_fft_ > buffer_fill_`: break
3. Shift buffer: move unconsumed samples to front
   - Keep last `n_fft_ - hop_length_` samples (overlap region)

---

### 3. COLA (Constant Overlap-Add) Guarantee

**Current Implementation**: Uses Hann window with 75% overlap (hop=n_fft/4)

**COLA Condition**:
```
sum(window[i + k*hop]) = constant, for all i
```

**Streaming Preservation**:
- Window coefficients are identical to batch mode
- Overlap-add happens in ISTFT, not STFT
- STFT streaming only affects **when** frames are computed, not **how**
- ✅ COLA is preserved as long as frame boundaries align with batch mode

---

### 4. Phase Continuity

**Analysis**:
- STFT computes independent FFTs per frame
- Phase is **not** accumulated across frames (unlike vocoder phase unwrapping)
- Each frame's phase is relative to its own time origin

**Conclusion**: ✅ No special handling needed for streaming STFT. Phase continuity is maintained by consistent frame alignment.

---

### 5. Center Padding Handling

**Batch Mode**: Reflects `n_fft/2` samples at start/end

**Streaming Mode**:
- **First chunk**: Prepend `n_fft/2` reflected samples from first chunk
- **Middle chunks**: No padding
- **Last chunk (Flush)**: Append `n_fft/2` reflected samples from last chunk

**Implementation**:
```cpp
int StreamingSTFT::ProcessChunk(const float* audio, int n_samples,
                                 float* output, int max_frames) {
    if (center_ && buffer_fill_ == 0) {
        // First chunk: add left padding
        int pad = n_fft_ / 2;
        for (int i = 0; i < pad; ++i) {
            input_buffer_[i] = audio[std::min(pad - i, n_samples - 1)];
        }
        buffer_fill_ = pad;
    }

    // Append audio
    std::memcpy(&input_buffer_[buffer_fill_], audio, n_samples * sizeof(float));
    buffer_fill_ += n_samples;

    // Extract frames...
}
```

---

### 6. Integration with Inference::Process()

**Current Flow**:
```
Process() → ComputeSTFT() → [allocate full buffer] → compute_stft()
```

**Streaming Flow**:
```
Process() → StreamingSTFT::ProcessChunk() → [yield frames incrementally]
         → PrepareModelInput() → RunModel() → StreamingISTFT::ProcessChunk()
```

**Key Changes**:
- Replace `stft_outputs[ch].resize(...)` with fixed-size ring buffer
- Process model in chunks of `chunk_size` frames
- Accumulate ISTFT output incrementally

---

### 7. StreamingISTFT Design

```cpp
class StreamingISTFT {
public:
    StreamingISTFT(int n_fft, int hop_length, int win_length, bool center = true);

    // Process STFT frames, returns reconstructed audio samples
    int ProcessFrames(const float* stft_frames, int n_frames, float* output);

    // Flush overlap-add buffer
    int Flush(float* output);

private:
    int n_fft_, hop_length_, win_length_;
    bool center_;

    // Overlap-add accumulator
    std::vector<float> ola_buffer_;      // Size: n_fft
    std::vector<float> window_sum_;      // Size: n_fft
    int ola_offset_;                     // Current write position

    // Window coefficients
    std::vector<float> window_padded_;

    // FFT workspace
    stft::STFTBuffer fft_buffer_;

    // Frame counter
    int frames_processed_;
};
```

**Overlap-Add Strategy**:
- Maintain circular buffer of size `n_fft`
- For each frame:
  1. IFFT → time-domain frame
  2. Add `frame[0:hop_length]` to `ola_buffer_[ola_offset_:ola_offset_+hop_length]`
  3. Output `ola_buffer_[ola_offset_:ola_offset_+hop_length]`
  4. Shift: `ola_offset_ = (ola_offset_ + hop_length) % n_fft`

---

### 8. Memory Comparison

**Batch Mode** (3-minute stereo, 44.1kHz, n_fft=4096, hop=1024):
```
n_samples = 3 * 60 * 44100 = 7,938,000
n_frames = 7,938,000 / 1024 ≈ 7,752
STFT buffer per channel = 2049 * 7752 * 2 * 4 bytes = 127 MB
Total (2 channels) = 254 MB
```

**Streaming Mode** (chunk_size=256 frames):
```
Input buffer = (4096 + 255 * 1024) * 4 bytes = 1.04 MB
STFT buffer = 2049 * 256 * 2 * 4 bytes = 4.2 MB
OLA buffer = 4096 * 4 bytes = 16 KB
Total per channel = 5.3 MB
Total (2 channels) = 10.6 MB
```

**Reduction**: 254 MB → 10.6 MB (**24x smaller**)

---

### 9. Implementation Complexity

**Difficulty**: Medium

**Components to Implement**:
1. `StreamingSTFT` class (~150 lines)
   - Buffer management: 40 lines
   - Frame extraction loop: 30 lines
   - Center padding logic: 30 lines
   - Flush handling: 20 lines

2. `StreamingISTFT` class (~120 lines)
   - Overlap-add accumulator: 50 lines
   - Circular buffer logic: 30 lines
   - Flush handling: 20 lines

3. `Inference::Process()` refactor (~80 lines)
   - Replace batch STFT calls: 20 lines
   - Chunk loop integration: 40 lines
   - Output accumulation: 20 lines

**Total**: ~350 lines of new/modified code

**Testing Requirements**:
- Unit tests: Verify streaming output matches batch output (bit-exact)
- Edge cases: Empty input, single frame, non-aligned chunk sizes
- Performance: Measure latency reduction vs batch mode

---

### 10. Pseudo-Code Example

```cpp
// Streaming STFT usage
StreamingSTFT stft(4096, 1024, 4096, true);
std::vector<float> output_frames(2049 * 256 * 2);  // 256 frames buffer

int total_frames = 0;
for (const auto& chunk : audio_chunks) {
    int n_frames = stft.ProcessChunk(chunk.data(), chunk.size(),
                                      output_frames.data(), 256);

    // Process frames with model
    ProcessModelChunk(output_frames.data(), n_frames);
    total_frames += n_frames;
}

// Flush remaining
int final_frames = stft.Flush(output_frames.data(), 256);
ProcessModelChunk(output_frames.data(), final_frames);
```

---

### 11. Validation Strategy

**Correctness Test**:
```cpp
// Generate test signal
std::vector<float> audio = GenerateSineWave(44100 * 10);  // 10 seconds

// Batch mode
auto batch_stft = ComputeSTFTBatch(audio);

// Streaming mode
StreamingSTFT stream_stft(4096, 1024, 4096, true);
std::vector<float> stream_output;
for (int i = 0; i < audio.size(); i += 44100) {  // 1-second chunks
    int chunk_size = std::min(44100, (int)audio.size() - i);
    // ... accumulate frames
}

// Compare
assert(batch_stft.size() == stream_output.size());
for (int i = 0; i < batch_stft.size(); ++i) {
    assert(std::abs(batch_stft[i] - stream_output[i]) < 1e-5);
}
```

---

## Summary

**Architecture**: Sliding window with incremental frame extraction

**Key Benefits**:
- 24x memory reduction (254 MB → 10.6 MB for 3-min audio)
- Enables real-time processing (latency = chunk_size * hop_length / sample_rate)
- Maintains bit-exact compatibility with batch mode

**COLA/Phase**: ✅ Preserved (no special handling needed)

**Implementation Effort**: ~350 lines, medium complexity

**Next Steps**:
1. Implement `StreamingSTFT` class in `src/streaming_stft.h`
2. Implement `StreamingISTFT` class
3. Refactor `Inference::Process()` to use streaming classes
4. Add unit tests for correctness validation
