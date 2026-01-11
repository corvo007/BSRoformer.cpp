#include "mel_band_roformer/inference.h"
#include "mel_band_roformer/audio.h"
#include <iostream>
#include <string>
#include <chrono>
#include <cstdlib>

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " <model.gguf> <input.wav> <output.wav> [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --chunk-size <N>   Chunk size in samples (default: from model, fallback 352800)" << std::endl;
    std::cerr << "  --overlap <N>      Number of overlaps for crossfade (default: from model, fallback 2)" << std::endl;
    std::cerr << "  --help, -h         Show this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    // Default values (will be overridden by model defaults if not explicitly set)
    int chunk_size = -1;  // -1 means use model default
    int num_overlap = -1; // -1 means use model default
    bool chunk_size_set = false;
    bool num_overlap_set = false;
    
    // Check for help flag first
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string input_path = argv[2];
    std::string output_path = argv[3];
    
    // Parse optional arguments
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--chunk-size" && i + 1 < argc) {
            try {
                chunk_size = std::stoi(argv[++i]);
                if (chunk_size <= 0) {
                     std::cerr << "Error: chunk-size must be a positive integer" << std::endl;
                     return 1;
                }
                chunk_size_set = true;
            } catch (...) {
                std::cerr << "Error: invalid chunk-size" << std::endl;
                return 1;
            }
        } else if (arg == "--overlap" && i + 1 < argc) {
            try {
                num_overlap = std::stoi(argv[++i]);
                if (num_overlap < 1) {
                    std::cerr << "Error: overlap must be at least 1" << std::endl;
                    return 1;
                }
                num_overlap_set = true;
             } catch (...) {
                std::cerr << "Error: invalid overlap" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        std::cout << "Initializing MelBandRoformer..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        Inference engine(model_path);
        
        // Use model defaults if not explicitly set by user
        if (!chunk_size_set) {
            chunk_size = engine.GetDefaultChunkSize();
        }
        if (!num_overlap_set) {
            num_overlap = engine.GetDefaultNumOverlap();
        }
        
        std::cout << "Loading audio: " << input_path << std::endl;
        AudioBuffer input_audio = AudioFile::Load(input_path);
        
        std::cout << "Audio loaded: " << input_audio.samples << " samples, " 
                  << input_audio.channels << " channels, " 
                  << input_audio.sampleRate << " Hz" << std::endl;

        // 1. Check Sample Rate
        int required_sr = engine.GetSampleRate();
        std::cout << "Model expects sample rate: " << required_sr << " Hz" << std::endl;

        if (input_audio.sampleRate != required_sr) {
            throw std::runtime_error("Input audio sample rate must be " + std::to_string(required_sr) + 
                                     " Hz. Current: " + std::to_string(input_audio.sampleRate));
        }

        // 2. Check Channels & Auto-Expand Mono
        if (input_audio.channels == 1) {
             std::cout << "[Info] Input is Mono. Expanding to Stereo..." << std::endl;
             std::vector<float> stereo_data(input_audio.samples * 2);
             for(size_t i=0; i<input_audio.samples; ++i) {
                 stereo_data[i*2 + 0] = input_audio.data[i];
                 stereo_data[i*2 + 1] = input_audio.data[i];
             }
             input_audio.data = std::move(stereo_data);
             input_audio.channels = 2;
             input_audio.samples *= 2;
        } else if (input_audio.channels != 2) {
             // We can either reject or try to process first 2 channels? 
             // Ideally reject to be safer, or warn.
             throw std::runtime_error("Input audio must be Stereo (2 channels) or Mono (1 channel). Current: " + std::to_string(input_audio.channels));
        }

        std::cout << "Processing with chunk_size=" << chunk_size 
                  << ", overlap=" << num_overlap << std::endl;
        auto process_start = std::chrono::high_resolution_clock::now();
        
        // Progress Bar Callback
        auto progress_callback = [](float progress) {
            int barWidth = 50;
            std::cout << "[";
            int pos = barWidth * progress;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << " %\r";
            std::cout.flush();
        };

        std::vector<std::vector<float>> output_stems = engine.Process(input_audio.data, chunk_size, num_overlap, progress_callback);

        // Clear progress line
        std::cout << std::string(70, ' ') << "\r";

        auto process_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = process_end - process_start;
        std::cout << "Processed in " << diff.count() << " seconds." << std::endl;
        
        int num_stems = output_stems.size();
        std::cout << "Model returned " << num_stems << " stems." << std::endl;

        for (int i = 0; i < num_stems; ++i) {
            // Prepare output filename
            std::string current_output_path = output_path;
            if (num_stems > 1) {
                // Insert _stem_i before extension
                size_t dot_pos = output_path.find_last_of(".");
                if (dot_pos != std::string::npos) {
                    current_output_path = output_path.substr(0, dot_pos) + "_stem_" + std::to_string(i) + output_path.substr(dot_pos);
                } else {
                    current_output_path = output_path + "_stem_" + std::to_string(i);
                }
            }

            // Prepare AudioBuffer
            AudioBuffer output_audio_buf;
            output_audio_buf.data = std::move(output_stems[i]); // Move to avoid copy
            output_audio_buf.channels = 2; // Output is always stereo
            output_audio_buf.sampleRate = required_sr;
            output_audio_buf.samples = output_audio_buf.data.size();
            
            std::cout << "Saving output stem " << i << ": " << current_output_path << std::endl;
            AudioFile::Save(current_output_path, output_audio_buf);
        }
        
        std::cout << "Done!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
