#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include "test_common.h"
#include "bs_roformer/inference.h"
#include "bs_roformer/audio.h"

static float MaxAbs(const std::vector<float>& v) {
    float m = 0.0f;
    for (float x : v) m = std::max(m, std::fabs(x));
    return m;
}

static std::string GetTestAudioPath() {
    if (const char* env = std::getenv("BSR_TEST_AUDIO_PATH")) {
        if (env && *env) return env;
    }
    // Default test fixture used during local dev (not committed; test will skip if missing)
    return "test_song_64s_8s.wav";
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Test: Inference Offset Stability (multi-chunk)" << std::endl;
    std::cout << "========================================" << std::endl;

    std::string model_path = GetModelPath();
    std::string audio_path = GetTestAudioPath();
    if (argc > 1) model_path = argv[1];
    if (argc > 2) audio_path = argv[2];

    if (!PathExists(model_path)) {
        TEST_SKIP("Model file not found: " + model_path + " (set BSR_MODEL_PATH)");
    }
    if (!PathExists(audio_path)) {
        TEST_SKIP("Test audio not found: " + audio_path + " (set BSR_TEST_AUDIO_PATH)");
    }

    // Read model settings
    Inference probe(model_path);
    const int chunk_frames = probe.GetDefaultChunkSize();
    const int required_sr = probe.GetSampleRate();
    TEST_ASSERT(chunk_frames > 0, "Model chunk_size must be positive");
    TEST_ASSERT(required_sr > 0, "Model sample rate must be positive");

    // Load a chunk that is known to produce non-trivial output when processed as the first chunk.
    AudioBuffer buf = AudioFile::Load(audio_path);
    TEST_ASSERT(buf.channels == 1 || buf.channels == 2, "Audio must be mono or stereo");
    TEST_ASSERT(static_cast<int>(buf.sampleRate) == required_sr,
                "Audio sample rate mismatch (expected " + std::to_string(required_sr) + ")");

    std::vector<float> stereo;
    if (buf.channels == 1) {
        const size_t frames = buf.samples;
        stereo.resize(frames * 2);
        for (size_t i = 0; i < frames; ++i) {
            float x = buf.data[i];
            stereo[i * 2 + 0] = x;
            stereo[i * 2 + 1] = x;
        }
    } else {
        stereo = std::move(buf.data);
    }

    const int64_t frames_in = static_cast<int64_t>(stereo.size() / 2);
    TEST_ASSERT(frames_in >= chunk_frames,
                "Test audio too short; need >= chunk_size frames (" + std::to_string(chunk_frames) + ")");

    std::vector<float> chunk(static_cast<size_t>(chunk_frames) * 2, 0.0f);
    std::copy(stereo.begin(), stereo.begin() + static_cast<std::ptrdiff_t>(chunk.size()), chunk.begin());

    // Baseline: process the chunk as chunk0 (fresh engine instance).
    Inference engine_base(model_path);
    auto base_stems = engine_base.Process(chunk, chunk_frames, /*num_overlap=*/1);
    TEST_ASSERT(!base_stems.empty(), "Baseline inference returned empty output");
    TEST_ASSERT(base_stems[0].size() == chunk.size(), "Baseline output size mismatch");

    const float base_max = MaxAbs(base_stems[0]);
    std::cout << "Baseline stem0 max_abs=" << base_max << std::endl;
    TEST_ASSERT(base_max > 1e-3f,
                "Baseline output unexpectedly silent; use a vocal-heavy test audio (set BSR_TEST_AUDIO_PATH)");

    // Offset test: prepend one full chunk of silence, then the same chunk. The second chunk should match baseline.
    std::vector<float> two_chunks(static_cast<size_t>(chunk_frames) * 2 * 2, 0.0f);
    std::copy(chunk.begin(), chunk.end(), two_chunks.begin() + static_cast<std::ptrdiff_t>(chunk.size()));

    Inference engine_off(model_path);
    auto off_stems = engine_off.Process(two_chunks, chunk_frames, /*num_overlap=*/1);
    TEST_ASSERT(!off_stems.empty(), "Offset inference returned empty output");
    TEST_ASSERT(off_stems[0].size() == two_chunks.size(), "Offset output size mismatch");

    std::vector<float> second_chunk(off_stems[0].begin() + static_cast<std::ptrdiff_t>(chunk.size()), off_stems[0].end());
    const float off_max = MaxAbs(second_chunk);
    std::cout << "Offset stem0 max_abs(second chunk)=" << off_max << std::endl;

    // Ignore a tiny guard region at both ends to avoid edge-only differences caused by window/counter clamping
    // (only a few samples can differ when overlap=1 and the first/last chunk window adjustments differ).
    const int guard_frames = 32;
    TEST_ASSERT(chunk_frames > guard_frames * 2, "chunk_size too small for guard comparison");
    const size_t start = static_cast<size_t>(guard_frames) * 2;
    const size_t end = static_cast<size_t>(chunk_frames - guard_frames) * 2;
    TEST_ASSERT(end > start, "Invalid interior slice");

    bool pass = CompareAndReport(
        "offset-stability-stem0",
        base_stems[0].data() + start, end - start,
        second_chunk.data() + start, end - start,
        /*atol=*/5e-3f,
        /*rtol=*/5e-2f);
    if (!pass) {
        LOG_FAIL();
        return 1;
    }

    LOG_PASS();
    return 0;
}
