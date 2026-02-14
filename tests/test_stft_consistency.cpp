#include "test_common.h"
#include "../src/stft.h"
#include "../src/model.h"

int main(int argc, char** argv) {
    std::cout << "Test: STFT/ISTFT Consistency with PyTorch" << std::endl;

    // 1. Load Model to get parameters
    std::string model_path = GetModelPath();
    std::cout << "Loading model params from: " << model_path << std::endl;
    
    // We only need the model to read parameters (n_fft, etc.) from GGUF
    // We don't need to allocate the full graph or weights.
    BSRoformer model;
    try {
        model.Initialize(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        std::cerr << "Ensure BSR_MODEL_PATH is set correctly or bs_roformer.gguf exists." << std::endl;
        return 1;
    }
    
    int n_fft = model.GetNFFT();
    int hop_length = model.GetHopLength();
    int win_length = model.GetWinLength();
    
    std::cout << "STFT Params: n_fft=" << n_fft << ", hop_length=" << hop_length << ", win_length=" << win_length << std::endl;
    
    // 2. Load Data
    std::string data_dir = GetTestDataDir();
    std::cout << "Loading test data from: " << data_dir << std::endl;
    
    GoldenTensor input_audio(data_dir, "input_audio"); // [batch, channels, samples]
    GoldenTensor expected_stft(data_dir, "stft_raw"); // [batch, channels, freq, time, 2]
    GoldenTensor expected_istft(data_dir, "istft_raw"); // [batch, channels, samples]
    
    TEST_ASSERT_LOAD(input_audio, "input_audio");
    TEST_ASSERT_LOAD(expected_stft, "stft_raw");
    TEST_ASSERT_LOAD(expected_istft, "istft_raw");
    
    input_audio.PrintShape("Input Audio");
    expected_stft.PrintShape("Expected STFT");
    expected_istft.PrintShape("Expected ISTFT");
    
    int batch = input_audio.shape[0];
    int channels = input_audio.shape[1];
    int n_samples = input_audio.shape[2];
    
    int n_freq = n_fft / 2 + 1;
    int expected_n_frames = expected_stft.shape[3]; 

    // 3. Prepare Window
    std::vector<float> window(win_length);
    stft::hann_window(window.data(), win_length);
    
    bool all_passed = true;
    
    // 4. Test STFT
    std::cout << "\n=== Testing STFT ===" << std::endl;
    
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            // Extract input channel
            std::vector<float> in_channel(n_samples);
            for (int i = 0; i < n_samples; ++i) {
                // Determine index based on memory layout
                // input_audio.npy is F-contiguous [1, 2, 220500] => [220500, 2] in memory (interleaved)
                // Layout: L0, R0, L1, R1, ...
                // Index = (sample_idx * channels + channel_idx)
                size_t idx = ((size_t)b * n_samples + i) * channels + c;
                in_channel[i] = input_audio.data[idx];
            }
            
            // Diagnostic: print first few input values
            std::cout << "  Input[" << b << "," << c << "] first 5: ";
            for (int i = 0; i < 5; ++i) std::cout << in_channel[i] << " ";
            std::cout << std::endl;
            
            int n_frames_calc = 0;
            // Buffer for output. 
            // C++ output is [n_freq, n_frames, 2]
            std::vector<float> out_stft(n_freq * (expected_n_frames + 10) * 2); 
            
            stft::compute_stft(
                in_channel.data(), n_samples, n_fft, hop_length, win_length,
                window.data(), true, out_stft.data(), &n_frames_calc
            );
            
            if (n_frames_calc != expected_n_frames) {
                std::cerr << "  [Batch " << b << " Ch " << c << "] Frame mismatch: calc=" << n_frames_calc << ", expected=" << expected_n_frames << std::endl;
                all_passed = false;
                continue;
            }
            
            // Compare
            size_t channel_stft_size = n_freq * expected_n_frames * 2;
            size_t offset = b * channels * channel_stft_size + c * channel_stft_size;
            
            std::string name = "STFT_B" + std::to_string(b) + "_Ch" + std::to_string(c);
            if (!CompareAndReport(name, 
                                  expected_stft.data + offset, channel_stft_size,
                                  out_stft.data(), channel_stft_size, 1e-3f, 1e-2f)) {
                all_passed = false;
            }
        }
    }
    
    // 5. Test ISTFT
    std::cout << "\n=== Testing ISTFT ===" << std::endl;
    
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
             size_t channel_stft_size = n_freq * expected_n_frames * 2;
             size_t offset = b * channels * channel_stft_size + c * channel_stft_size;
             
             // Input: expected_stft.data + offset
             std::vector<float> out_audio(n_samples + n_fft); // Buffer slightly larger
             
             // We pass n_samples as expected length
             stft::compute_istft(
                 expected_stft.data + offset,
                 n_freq, expected_n_frames, n_fft, hop_length, win_length,
                 window.data(), true, n_samples, out_audio.data()
             );
             
             // Verify against expected_istft
             size_t audio_offset = b * channels * n_samples + c * n_samples;
             
             std::string name = "ISTFT_B" + std::to_string(b) + "_Ch" + std::to_string(c);
             if (!CompareAndReport(name,
                                   expected_istft.data + audio_offset, n_samples,
                                   out_audio.data(), n_samples, 1e-4f, 1e-3f)) {
                 all_passed = false;                     
             }
        }
    }
    
    if (all_passed) {
        LOG_PASS();
        return 0;
    } else {
        LOG_FAIL();
        return 1;
    }
}
