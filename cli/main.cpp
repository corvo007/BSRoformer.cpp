#include "bs_roformer/inference.h"
#include "bs_roformer/audio.h"
#include "dr_libs/dr_wav.h"
#include <iostream>
#include <string>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

template<typename T>
class ThreadSafeQueue {
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_push_, cv_pop_;
    size_t max_size_;
public:
    explicit ThreadSafeQueue(size_t max_size = 8) : max_size_(max_size) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_push_.wait(lock, [this]{ return queue_.size() < max_size_; });
        queue_.push(std::move(item));
        cv_pop_.notify_one();
    }
    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_pop_.wait(lock, [this]{ return !queue_.empty(); });
        item = std::move(queue_.front());
        queue_.pop();
        cv_push_.notify_one();
        return true;
    }
};

struct InputChunk {
    std::vector<float> data;
    bool is_end = false;
};

struct OutputChunk {
    std::vector<std::vector<float>> stems;
    bool is_end = false;
};

static std::string MakeStemOutputPath(const std::string& output_path, int stem_idx, int num_stems) {
    if (num_stems <= 1) return output_path;

    size_t dot_pos = output_path.find_last_of(".");
    if (dot_pos != std::string::npos) {
        return output_path.substr(0, dot_pos) + "_stem_" + std::to_string(stem_idx) + output_path.substr(dot_pos);
    }
    return output_path + "_stem_" + std::to_string(stem_idx);
}

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " <model.gguf> <input.wav> <output.wav> [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --chunk-size <N>   Chunk size in samples (default: from model, fallback 352800)" << std::endl;
    std::cerr << "  --overlap <N>      Number of overlaps for crossfade (default: from model, fallback 2)" << std::endl;
    std::cerr << "  --no-stream        Disable streaming I/O (debug only; uses more RAM)" << std::endl;
    std::cerr << "  --no-pipeline      Disable pipelined streaming inference (debug only)" << std::endl;
    std::cerr << "  --help, -h         Show this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    // Default values (will be overridden by model defaults if not explicitly set)
    int chunk_size = -1;  // -1 means use model default
    int num_overlap = -1; // -1 means use model default
    bool chunk_size_set = false;
    bool num_overlap_set = false;
    bool use_streaming_io = true; // Default: streaming
    bool use_pipelined_stream = true; // Default: pipelined streaming inference
    
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
        } else if (arg == "--no-stream") {
            use_streaming_io = false;
        } else if (arg == "--no-pipeline") {
            use_pipelined_stream = false;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        std::cout << "Initializing BSRoformer..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        Inference engine(model_path);
        
        // Use model defaults if not explicitly set by user
        if (!chunk_size_set) {
            chunk_size = engine.GetDefaultChunkSize();
        }
        if (!num_overlap_set) {
            num_overlap = engine.GetDefaultNumOverlap();
        }
        
        int required_sr = engine.GetSampleRate();
        std::cout << "Model expects sample rate: " << required_sr << " Hz" << std::endl;

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

        std::vector<std::vector<float>> output_stems;
        if (!use_streaming_io) {
            std::cout << "[Info] Streaming disabled: loading full WAV into memory..." << std::endl;
            AudioBuffer input_audio = AudioFile::Load(input_path);

            std::cout << "Audio loaded: " << input_audio.samples << " samples, "
                      << input_audio.channels << " channels, "
                      << input_audio.sampleRate << " Hz" << std::endl;

            if (input_audio.sampleRate != static_cast<unsigned int>(required_sr)) {
                throw std::runtime_error("Input audio sample rate must be " + std::to_string(required_sr) +
                                         " Hz. Current: " + std::to_string(input_audio.sampleRate));
            }

            if (input_audio.channels == 1) {
                std::cout << "[Info] Input is Mono. Expanding to Stereo..." << std::endl;
                const size_t frames = input_audio.samples;
                std::vector<float> stereo_data(frames * 2);
                for (size_t i = 0; i < frames; ++i) {
                    stereo_data[i * 2 + 0] = input_audio.data[i];
                    stereo_data[i * 2 + 1] = input_audio.data[i];
                }
                input_audio.data = std::move(stereo_data);
                input_audio.channels = 2;
                input_audio.samples = input_audio.data.size();
            } else if (input_audio.channels != 2) {
                throw std::runtime_error("Input audio must be Stereo (2 channels) or Mono (1 channel). Current: " +
                                         std::to_string(input_audio.channels));
            }

            output_stems = engine.Process(input_audio.data, chunk_size, num_overlap, progress_callback);
        } else {
            std::cout << "[Info] Streaming I/O enabled (default)" << std::endl;
            if (!use_pipelined_stream) {
                std::cout << "[Info] Pipelined streaming inference disabled (--no-pipeline)" << std::endl;
            }
            drwav wav;
            if (!drwav_init_file(&wav, input_path.c_str(), nullptr)) {
                throw std::runtime_error("Failed to open audio file: " + input_path);
            }

            const unsigned int in_ch = wav.channels;
            const unsigned int in_sr = wav.sampleRate;
            const drwav_uint64 total_frames = wav.totalPCMFrameCount;

            std::cout << "Audio opened: " << (total_frames * in_ch) << " samples, "
                      << in_ch << " channels, "
                      << in_sr << " Hz" << std::endl;

            if (in_sr != static_cast<unsigned int>(required_sr)) {
                drwav_uninit(&wav);
                throw std::runtime_error("Input audio sample rate must be " + std::to_string(required_sr) +
                                         " Hz. Current: " + std::to_string(in_sr));
            }

            if (!(in_ch == 1 || in_ch == 2)) {
                drwav_uninit(&wav);
                throw std::runtime_error("Input audio must be Stereo (2 channels) or Mono (1 channel). Current: " +
                                         std::to_string(in_ch));
            }

            const int stems = engine.GetNumStems();
            std::cout << "Model reports " << stems << " stems." << std::endl;

            // Open writers (one per stem)
            drwav_data_format format{};
            format.container = drwav_container_riff;
            format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
            format.channels = 2; // output is always stereo
            format.sampleRate = required_sr;
            format.bitsPerSample = 32;

            std::vector<drwav> writers(stems);
            std::vector<std::string> stem_paths(stems);
            for (int s = 0; s < stems; ++s) {
                stem_paths[s] = MakeStemOutputPath(output_path, s, stems);
                if (!drwav_init_file_write(&writers[s], stem_paths[s].c_str(), &format, nullptr)) {
                    for (int j = 0; j < s; ++j) drwav_uninit(&writers[j]);
                    drwav_uninit(&wav);
                    throw std::runtime_error("Failed to open file for writing: " + stem_paths[s]);
                }
            }

            auto stream = engine.CreateStream(chunk_size, num_overlap, use_pipelined_stream);
            const drwav_uint64 read_frames = static_cast<drwav_uint64>(chunk_size);

            ThreadSafeQueue<InputChunk> input_queue;
            ThreadSafeQueue<OutputChunk> output_queue;
            std::atomic<drwav_uint64> frames_read_total{0};

            // Reader thread
            std::thread reader([&]() {
                std::vector<float> read_buf(static_cast<size_t>(read_frames) * in_ch);
                while (true) {
                    drwav_uint64 got = drwav_read_pcm_frames_f32(&wav, read_frames, read_buf.data());
                    if (got == 0) break;

                    InputChunk chunk;
                    if (in_ch == 1) {
                        chunk.data.resize(static_cast<size_t>(got) * 2);
                        for (drwav_uint64 i = 0; i < got; ++i) {
                            float x = read_buf[static_cast<size_t>(i)];
                            chunk.data[static_cast<size_t>(i) * 2 + 0] = x;
                            chunk.data[static_cast<size_t>(i) * 2 + 1] = x;
                        }
                    } else {
                        chunk.data.assign(read_buf.begin(), read_buf.begin() + static_cast<size_t>(got) * in_ch);
                    }
                    input_queue.push(std::move(chunk));
                    frames_read_total += got;
                }
                InputChunk end; end.is_end = true;
                input_queue.push(std::move(end));
            });

            // Writer thread
            std::thread writer([&]() {
                while (true) {
                    OutputChunk chunk;
                    output_queue.pop(chunk);
                    if (chunk.is_end) break;

                    for (int s = 0; s < stems; ++s) {
                        if (chunk.stems.size() <= static_cast<size_t>(s)) continue;
                        drwav_write_pcm_frames(&writers[s], chunk.stems[s].size() / 2, chunk.stems[s].data());
                    }
                }
            });

            // Main processing loop
            float last_progress = -0.05f;
            while (true) {
                InputChunk chunk;
                input_queue.pop(chunk);
                if (chunk.is_end) break;

                auto out = engine.ProcessStream(*stream, chunk.data);

                OutputChunk out_chunk;
                out_chunk.stems = std::move(out);
                output_queue.push(std::move(out_chunk));

                if (total_frames > 0) {
                    float progress = static_cast<float>(frames_read_total.load()) / static_cast<float>(total_frames);
                    if (progress - last_progress >= 0.05f) {
                        progress_callback(progress);
                        last_progress = progress;
                    }
                }
            }

            auto tail = engine.FinalizeStream(*stream);
            if (!tail.empty()) {
                OutputChunk tail_chunk;
                tail_chunk.stems = std::move(tail);
                output_queue.push(std::move(tail_chunk));
            }

            OutputChunk end; end.is_end = true;
            output_queue.push(std::move(end));

            reader.join();
            writer.join();

            // Clear progress line
            std::cout << std::string(70, ' ') << "\r";

            drwav_uninit(&wav);
            for (int s = 0; s < stems; ++s) {
                drwav_uninit(&writers[s]);
                std::cout << "Saved output stem " << s << ": " << stem_paths[s] << std::endl;
            }
        }

        // Clear progress line
        if (!use_streaming_io) {
            std::cout << std::string(70, ' ') << "\r";
        }

        auto process_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = process_end - process_start;
        std::cout << "Processed in " << diff.count() << " seconds." << std::endl;
        
        if (!use_streaming_io) {
            int num_stems = static_cast<int>(output_stems.size());
            std::cout << "Model returned " << num_stems << " stems." << std::endl;

            for (int i = 0; i < num_stems; ++i) {
                std::string current_output_path = MakeStemOutputPath(output_path, i, num_stems);

                AudioBuffer output_audio_buf;
                output_audio_buf.data = std::move(output_stems[i]); // Move to avoid copy
                output_audio_buf.channels = 2; // Output is always stereo
                output_audio_buf.sampleRate = required_sr;
                output_audio_buf.samples = output_audio_buf.data.size();

                std::cout << "Saving output stem " << i << ": " << current_output_path << std::endl;
                AudioFile::Save(current_output_path, output_audio_buf);
            }
        }
        
        std::cout << "Done!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
