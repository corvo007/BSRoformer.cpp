#include <iostream>
#include <vector>
#include <cmath>
#include "bs_roformer/audio.h"

int main() {
    std::cout << "Test: Audio I/O with dr_wav" << std::endl;
    
    const std::string test_file = "test_tone.wav";
    const int sample_rate = 44100;
    const int duration_sec = 1;
    const int channels = 2;
    const int total_samples = sample_rate * duration_sec * channels;
    
    // 1. Generate Stereo Sine Wave (440Hz Left, 880Hz Right)
    AudioBuffer gen_buffer;
    gen_buffer.channels = channels;
    gen_buffer.sampleRate = sample_rate;
    gen_buffer.samples = total_samples;
    gen_buffer.data.resize(total_samples);
    
    for (int i = 0; i < sample_rate * duration_sec; ++i) {
        float t = (float)i / sample_rate;
        float val_left = std::sin(2.0f * 3.14159f * 440.0f * t);
        float val_right = std::sin(2.0f * 3.14159f * 880.0f * t);
        
        gen_buffer.data[i * 2 + 0] = val_left;
        gen_buffer.data[i * 2 + 1] = val_right;
    }
    
    // 2. Write to File
    std::cout << "Writing " << test_file << "..." << std::endl;
    try {
        AudioFile::Save(test_file, gen_buffer);
    } catch (const std::exception& e) {
        std::cerr << "Error writing: " << e.what() << std::endl;
        return 1;
    }
    
    // 3. Read Back
    std::cout << "Reading " << test_file << "..." << std::endl;
    AudioBuffer read_buffer;
    try {
        read_buffer = AudioFile::Load(test_file);
    } catch (const std::exception& e) {
        std::cerr << "Error reading: " << e.what() << std::endl;
        return 1;
    }
    
    // 4. Verify
    if (read_buffer.sampleRate != sample_rate) {
        std::cerr << "FAILED: Sample rate mismatch " << read_buffer.sampleRate << " != " << sample_rate << std::endl;
        return 1;
    }
    
    if (read_buffer.channels != channels) {
         std::cerr << "FAILED: Channel count mismatch " << read_buffer.channels << " != " << channels << std::endl;
         return 1;
    }
    
    float max_diff = 0.0f;
    for (size_t i = 0; i < read_buffer.samples; ++i) {
        float diff = std::abs(read_buffer.data[i] - gen_buffer.data[i]);
        if (diff > max_diff) max_diff = diff;
    }
    
    std::cout << "Max diff: " << max_diff << std::endl;
    
    if (max_diff > 1e-4) {
        std::cerr << "FAILED: Data mismatch (diff > 1e-4)" << std::endl;
        return 1;
    }
    
    std::cout << "PASSED: Audio I/O Verified." << std::endl;
    return 0;
}
