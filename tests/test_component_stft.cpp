#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "../src/stft.h"

int main() {
    std::cout << "Test: Component STFT/ISTFT" << std::endl;
    
    // Parameters
    const int sample_rate = 44100;
    const int n_fft = 2048;
    const int hop_length = 441;
    const int win_length = 2048;
    const int n_freq = n_fft / 2 + 1;
    const int n_samples = 44100 * 2; // 2 seconds
    
    // 1. Generate Signal (Sine wave mixture)
    std::vector<float> input(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        input[i] = std::sin(2.0f * M_PI * 440.0f * t) + 
                   0.5f * std::sin(2.0f * M_PI * 880.0f * t);
    }
    
    // 2. Generate Window
    std::vector<float> window(win_length);
    stft::hann_window(window.data(), win_length);
    
    // 3. Compute STFT
    int n_frames = 0;
    // Estimate size: n_freq * estimated_frames * 2, give some buffer
    std::vector<float> stft_out(n_freq * 500 * 2); 
    
    stft::compute_stft(
        input.data(),
        n_samples,
        n_fft,
        hop_length,
        win_length,
        window.data(),
        true, // center
        stft_out.data(),
        &n_frames
    );
    
    std::cout << "STFT Computed: " << n_frames << " frames" << std::endl;
    
    if (n_frames == 0) {
        std::cerr << "Failed: 0 frames" << std::endl;
        return 1;
    }
    
    // 4. Compute ISTFT
    std::vector<float> output(n_samples);
    
    stft::compute_istft(
        stft_out.data(),
        n_freq,
        n_frames,
        n_fft,
        hop_length,
        win_length,
        window.data(),
        true, // center
        n_samples,
        output.data()
    );
    
    // 5. Verify Reconstruction (MSE/MAE)
    float max_diff = 0.0f;
    float mae = 0.0f;
    
    for (int i = 0; i < n_samples; ++i) {
        float diff = std::abs(input[i] - output[i]);
        if (diff > max_diff) max_diff = diff;
        mae += diff;
    }
    mae /= n_samples;
    
    std::cout << "Reconstruction Error:" << std::endl;
    std::cout << "  Max Diff: " << max_diff << std::endl;
    std::cout << "  MAE:      " << mae << std::endl;
    
    // STFT/ISTFT with Hann window and overlap >= 50% should be near perfect
    // COLA constraint check: 2048/441 = ~4.6 overlaps, excellent.
    
    if (max_diff > 1e-4) {
        std::cerr << "FAILED: Reconstruction error too high (> 1e-4)" << std::endl;
        return 1;
    }
    
    std::cout << "PASSED" << std::endl;
    return 0;
}
