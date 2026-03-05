#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include "test_common.h"
#include "bs_roformer/inference.h"

static float MaxAbs(const std::vector<float>& v) {
    float m = 0.0f;
    for (float x : v) m = std::max(m, std::fabs(x));
    return m;
}

static std::vector<float> MakeSyntheticVoiceLikeStereo(int frames, int sample_rate, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const float two_pi = 6.2831853071795864769f;
    const float f0 = 140.0f;

    std::vector<float> out(static_cast<size_t>(frames) * 2, 0.0f);
    for (int i = 0; i < frames; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(sample_rate);
        float env = 1.0f;
        if (frames > 1) {
            // Hann envelope to avoid hard edges.
            env = 0.5f - 0.5f * std::cos(two_pi * static_cast<float>(i) / static_cast<float>(frames - 1));
        }

        float x = 0.0f;
        for (int h = 1; h <= 8; ++h) {
            x += (1.0f / static_cast<float>(h)) * std::sin(two_pi * f0 * static_cast<float>(h) * t);
        }

        // Keep well within [-1, 1] and add a touch of noise.
        x = 0.18f * env * x + 0.01f * dist(rng);

        out[static_cast<size_t>(i) * 2 + 0] = x;
        out[static_cast<size_t>(i) * 2 + 1] = x;
    }
    return out;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Test: Inference Repeatability" << std::endl;
    std::cout << "========================================" << std::endl;

    std::string model_path = GetModelPath();
    if (argc > 1) model_path = argv[1];

    if (!PathExists(model_path)) {
        TEST_SKIP("Model file not found: " + model_path + " (set BSR_MODEL_PATH)");
    }

    Inference engine(model_path);

    const int chunk_frames = engine.GetDefaultChunkSize();
    const int sr = engine.GetSampleRate();
    TEST_ASSERT(chunk_frames > 0, "Model default chunk size must be positive");
    TEST_ASSERT(sr > 0, "Model sample rate must be positive");

    auto input = MakeSyntheticVoiceLikeStereo(chunk_frames, sr, /*seed=*/123);

    auto out1 = engine.ProcessChunk(input);
    auto out2 = engine.ProcessChunk(input);

    TEST_ASSERT(!out1.empty(), "First inference returned empty output");
    TEST_ASSERT(out2.size() == out1.size(), "Second inference stem count mismatch");

    for (size_t s = 0; s < out1.size(); ++s) {
        TEST_ASSERT(out1[s].size() == out2[s].size(), "Output size mismatch between runs");

        const float e1 = MaxAbs(out1[s]);
        const float e2 = MaxAbs(out2[s]);
        std::cout << "Stem " << s << ": max_abs(run1)=" << e1 << " max_abs(run2)=" << e2 << std::endl;

        bool pass = CompareAndReport(
            "repeatability_stem_" + std::to_string(s),
            out1[s].data(), out1[s].size(),
            out2[s].data(), out2[s].size(),
            /*atol=*/5e-3f,
            /*rtol=*/5e-2f);

        if (!pass) {
            LOG_FAIL();
            return 1;
        }
    }

    LOG_PASS();
    return 0;
}

