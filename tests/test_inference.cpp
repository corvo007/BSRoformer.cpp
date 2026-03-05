#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cstdlib>
#include <algorithm>
#include "test_common.h"
#include "bs_roformer/inference.h"
#include "../src/utils.h"

/**
 * test_inference.cpp
 * 
 * Verifies Inference class against golden tensors from export_debug.py
 * Copied from tests_old/test_inference.cpp with env var support
 */

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Test: Inference Class Verification" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::string model_path = GetModelPath();
    std::string debug_dir = GetTestDataDir();
    
    if (argc > 1) model_path = argv[1];
    if (argc > 2) debug_dir = argv[2];

    if (!PathExists(model_path)) {
        TEST_SKIP("Model file not found: " + model_path + " (set BSR_MODEL_PATH)");
    }
    if (!PathExists(ActivationPath(debug_dir, "input_audio")) ||
        !PathExists(ActivationPath(debug_dir, "output_audio"))) {
        TEST_SKIP("Test data not found under: " + debug_dir + " (set BSR_TEST_DATA_DIR)");
    }
    
    try {
        // 1. Initialize Inference
        std::cout << "\n[1/3] Initializing Inference Engine..." << std::endl;
        Inference engine(model_path);
        
        // 2. Load Input Audio
        std::cout << "\n[2/3] Loading Input Audio..." << std::endl;
        auto [input_audio_ptr, input_audio_shape] = utils::load_activation(debug_dir, "input_audio");
        if (!input_audio_ptr) return 1;
        
        // Convert to vector (input_audio.npy is [batch, channels, samples] interleaved)
        // input_audio_shape: [1, 2, 132300]
        size_t total_samples = input_audio_shape[0] * input_audio_shape[1] * input_audio_shape[2];
        std::vector<float> input_audio(input_audio_ptr, input_audio_ptr + total_samples);
        
        // 3. Process
        std::cout << "\n[3/3] Processing Audio..." << std::endl;
        // Use ProcessChunk to verify raw model output without Overlap-Add windowing/padding
        // This matches the generation of output_audio.npy
        std::vector<std::vector<float>> output_stems = engine.ProcessChunk(input_audio);
        std::vector<float> output_audio = output_stems[0];

        // Smoke test new cancel callback path in Process()
        size_t smoke_samples = std::min<size_t>(input_audio.size(), static_cast<size_t>(16384 * 2));
        if (smoke_samples % 2 != 0) {
            smoke_samples -= 1;
        }
        if (smoke_samples >= 2) {
            std::vector<float> smoke_input(input_audio.begin(), input_audio.begin() + smoke_samples);
            auto cancel_false = []() { return false; };
            auto smoke_stems = engine.Process(smoke_input, 16384, 2, nullptr, cancel_false);
            if (smoke_stems.empty() || smoke_stems[0].empty()) {
                std::cerr << "Process() smoke test returned empty output" << std::endl;
                return 1;
            }
        }
        
        std::cout << "  Input size: " << input_audio.size() << std::endl;
        std::cout << "  Output size: " << output_audio.size() << std::endl;
        
        // Verify against output_audio.npy
        std::cout << "\n[Verification] Comparing against golden output..." << std::endl;
        auto [expected_output, expected_shape] = utils::load_activation(debug_dir, "output_audio");
        if (!expected_output) {
             std::cerr << "Golden output not found" << std::endl;
             return 1;
        }
        
        // expected_output: [batch=1, channels=2, samples=132300] (Planar/C-contiguous)
        // output_audio: interleaved [ch0, ch1, ch0, ch1...]
        
        int channels = 2;
        int samples = input_audio_shape[2]; // 132300
        
        float max_diff = 0.0f;
        float sum_diff = 0.0f;
        int valid_samples = 0;
        
        for (int i = 0; i < samples; ++i) {
            for (int ch = 0; ch < channels; ++ch) {
                // Expected: ch * samples + i
                float expected = expected_output[ch * samples + i];
                
                // Actual: i * channels + ch
                if (i * channels + ch >= output_audio.size()) continue;
                float actual = output_audio[i * channels + ch];
                
                float diff = std::abs(expected - actual);
                max_diff = std::max(max_diff, diff);
                sum_diff += diff;
                valid_samples++;
            }
        }
        
        if (valid_samples == 0) valid_samples = 1;

        std::cout << "  Max abs diff: " << max_diff << std::endl;
        std::cout << "  Mean abs diff: " << (sum_diff / valid_samples) << std::endl;
        
        bool pass = (sum_diff / valid_samples) < 0.1f;
        if (pass) std::cout << "PASSED" << std::endl;
        else std::cout << "FAILED" << std::endl;
        
        utils::free_npy_data(input_audio_ptr);
        utils::free_npy_data(expected_output);
        
        return pass ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
