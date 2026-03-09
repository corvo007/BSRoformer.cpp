#include "bs_roformer/inference.h"
#include "bs_roformer/audio.h"
#include "process_utils.h"
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
#include <filesystem>
#include <sstream>
#include <iomanip>

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
    return output_path + "_stem_" + std::to_string(stem_idx) + ".wav";
}

static std::filesystem::path CreateTempDir() {
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    std::ostringstream name;
    name << "bs_roformer_segments_" << stamp;

    std::filesystem::path dir = std::filesystem::temp_directory_path() / name.str();
    std::filesystem::create_directories(dir);
    return dir;
}

static bool FileExists(const std::filesystem::path& p) {
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}

static void StreamCopyAll(drwav& reader, drwav& writer) {
    constexpr drwav_uint64 kBlockFrames = 1u << 16;
    std::vector<float> buf(static_cast<size_t>(kBlockFrames) * 2);
    while (true) {
        drwav_uint64 got = drwav_read_pcm_frames_f32(&reader, kBlockFrames, buf.data());
        if (got == 0) break;
        drwav_write_pcm_frames(&writer, got, buf.data());
    }
}

static void StreamCopyAllButKeepTail(drwav& reader,
                                    drwav& writer,
                                    drwav_uint64 keep_tail_frames,
                                    std::vector<float>& out_tail /*stereo interleaved*/) {
    constexpr drwav_uint64 kBlockFrames = 1u << 16;
    if (keep_tail_frames == 0) {
        StreamCopyAll(reader, writer);
        out_tail.clear();
        return;
    }

    std::vector<float> buf(static_cast<size_t>(kBlockFrames) * 2);
    std::vector<float> pending;
    pending.reserve(static_cast<size_t>(keep_tail_frames + kBlockFrames) * 2);
    size_t pending_start = 0;

    auto pending_frames = [&]() -> drwav_uint64 {
        return static_cast<drwav_uint64>((pending.size() - pending_start) / 2);
    };

    while (true) {
        drwav_uint64 got = drwav_read_pcm_frames_f32(&reader, kBlockFrames, buf.data());
        if (got == 0) break;

        pending.insert(pending.end(), buf.begin(), buf.begin() + static_cast<size_t>(got) * 2);

        const drwav_uint64 have_frames = pending_frames();
        if (have_frames > keep_tail_frames) {
            const drwav_uint64 write_frames = have_frames - keep_tail_frames;
            drwav_write_pcm_frames(&writer, write_frames, pending.data() + pending_start);
            pending_start += static_cast<size_t>(write_frames) * 2;

            if (pending_start > static_cast<size_t>(kBlockFrames) * 8) {
                pending.erase(pending.begin(), pending.begin() + static_cast<std::vector<float>::difference_type>(pending_start));
                pending_start = 0;
            }
        }
    }

    out_tail.assign(pending.begin() + static_cast<std::vector<float>::difference_type>(pending_start), pending.end());
}

static void CrossfadeAndWrite(drwav& writer,
                             const std::vector<float>& a /*stereo interleaved*/,
                             const std::vector<float>& b /*stereo interleaved*/,
                             drwav_uint64 frames) {
    if (frames == 0) return;
    if (a.size() < static_cast<size_t>(frames) * 2 || b.size() < static_cast<size_t>(frames) * 2) {
        throw std::runtime_error("Crossfade buffers are smaller than requested frames");
    }

    std::vector<float> out(static_cast<size_t>(frames) * 2);
    const float denom = (frames > 1) ? static_cast<float>(frames - 1) : 1.0f;
    for (drwav_uint64 f = 0; f < frames; ++f) {
        const float alpha = (frames > 1) ? (static_cast<float>(f) / denom) : 1.0f;
        const float wa = 1.0f - alpha;
        const float wb = alpha;

        const size_t idx = static_cast<size_t>(f) * 2;
        out[idx + 0] = a[idx + 0] * wa + b[idx + 0] * wb;
        out[idx + 1] = a[idx + 1] * wa + b[idx + 1] * wb;
    }

    drwav_write_pcm_frames(&writer, frames, out.data());
}

static int DetectNumStemsFromSegmentOutputs(const std::filesystem::path& segment_output_base) {
    if (FileExists(segment_output_base)) return 1;

    for (int s = 1; s <= 16; ++s) {
        const auto stem0 = std::filesystem::path(MakeStemOutputPath(segment_output_base.string(), 0, s));
        const auto stemLast = std::filesystem::path(MakeStemOutputPath(segment_output_base.string(), s - 1, s));
        if (FileExists(stem0) && FileExists(stemLast)) {
            return s;
        }
    }

    throw std::runtime_error("Failed to detect stem output files from first segment");
}

static void MergeOneSegmentStem(const std::filesystem::path& seg_stem_path,
                               drwav& writer,
                               size_t seg_idx,
                               size_t seg_count,
                               drwav_uint64 overlap_frames,
                               std::vector<float>& prev_tail /*in/out*/) {
    drwav reader{};
    if (!drwav_init_file(&reader, seg_stem_path.string().c_str(), nullptr)) {
        throw std::runtime_error("Failed to open segment stem for reading: " + seg_stem_path.string());
    }

    const unsigned int ch = reader.channels;
    if (ch != 2) {
        drwav_uninit(&reader);
        throw std::runtime_error("Segment stem must be stereo (2 channels): " + seg_stem_path.string());
    }

    if (writer.sampleRate != 0 && reader.sampleRate != writer.sampleRate) {
        drwav_uninit(&reader);
        throw std::runtime_error("Sample rate mismatch while merging segment stems");
    }

    const bool is_first = (seg_idx == 0);
    const bool is_last = (seg_idx + 1 >= seg_count);

    if (overlap_frames == 0 || seg_count <= 1) {
        StreamCopyAll(reader, writer);
        drwav_uninit(&reader);
        prev_tail.clear();
        return;
    }

    if (is_first) {
        StreamCopyAllButKeepTail(reader, writer, overlap_frames, prev_tail);
        drwav_uninit(&reader);
        if (prev_tail.size() != static_cast<size_t>(overlap_frames) * 2) {
            throw std::runtime_error("First segment is shorter than segment overlap; reduce overlap or segment length");
        }
        return;
    }

    if (prev_tail.size() != static_cast<size_t>(overlap_frames) * 2) {
        drwav_uninit(&reader);
        throw std::runtime_error("Previous tail size mismatch during segment merge");
    }

    std::vector<float> head(static_cast<size_t>(overlap_frames) * 2);
    drwav_uint64 got_head = drwav_read_pcm_frames_f32(&reader, overlap_frames, head.data());
    if (got_head != overlap_frames) {
        drwav_uninit(&reader);
        throw std::runtime_error("Segment is shorter than segment overlap; reduce overlap or segment length");
    }

    CrossfadeAndWrite(writer, prev_tail, head, overlap_frames);

    if (is_last) {
        StreamCopyAll(reader, writer);
        drwav_uninit(&reader);
        prev_tail.clear();
        return;
    }

    StreamCopyAllButKeepTail(reader, writer, overlap_frames, prev_tail);
    drwav_uninit(&reader);
    if (prev_tail.size() != static_cast<size_t>(overlap_frames) * 2) {
        throw std::runtime_error("Segment tail is shorter than segment overlap; reduce overlap or segment length");
    }
}

static int RunSegmentedMultiprocess(const std::filesystem::path& exe_path,
                                   const std::string& model_path,
                                   const std::string& input_path,
                                   const std::string& output_path,
                                   bool chunk_size_set,
                                   int chunk_size,
                                   bool num_overlap_set,
                                   int num_overlap,
                                   bool use_pipelined_stream,
                                   bool use_io_threads,
                                   int segment_minutes,
                                   int segment_overlap_seconds,
                                   bool keep_temps) {
    drwav in_wav{};
    if (!drwav_init_file(&in_wav, input_path.c_str(), nullptr)) {
        throw std::runtime_error("Failed to open audio file: " + input_path);
    }

    const unsigned int in_sr = in_wav.sampleRate;
    const drwav_uint64 total_frames = in_wav.totalPCMFrameCount;
    drwav_uninit(&in_wav);

    if (segment_minutes <= 0) {
        throw std::runtime_error("segment-minutes must be a positive integer");
    }
    if (segment_overlap_seconds < 0) {
        throw std::runtime_error("segment-overlap-seconds must be >= 0");
    }

    const drwav_uint64 segment_frames = static_cast<drwav_uint64>(segment_minutes) * 60ull * static_cast<drwav_uint64>(in_sr);
    const drwav_uint64 overlap_frames = static_cast<drwav_uint64>(segment_overlap_seconds) * static_cast<drwav_uint64>(in_sr);

    if (segment_frames == 0) {
        throw std::runtime_error("Invalid segment length (0 frames)");
    }
    if (overlap_frames >= segment_frames && segment_overlap_seconds > 0) {
        throw std::runtime_error("segment overlap must be shorter than segment length");
    }

    const size_t seg_count = static_cast<size_t>((total_frames + segment_frames - 1) / segment_frames);
    if (seg_count <= 1) {
        std::cout << "[Info] Input shorter than one segment; running single-process inference in a child process." << std::endl;
        std::vector<std::string> child_args;
        child_args.reserve(16);
        child_args.push_back(model_path);
        child_args.push_back(input_path);
        child_args.push_back(output_path);
        if (chunk_size_set) {
            child_args.push_back("--chunk-size");
            child_args.push_back(std::to_string(chunk_size));
        }
        if (num_overlap_set) {
            child_args.push_back("--overlap");
            child_args.push_back(std::to_string(num_overlap));
        }
        if (!use_io_threads) {
            child_args.push_back("--no-io-threads");
        }
        if (!use_pipelined_stream) {
            child_args.push_back("--no-pipeline");
        }

        return cli_process::SpawnChildAndWait(exe_path, child_args);
    }

    std::cout << "[Info] Multiprocess segmentation enabled: " << seg_count << " segments of "
              << segment_minutes << " minutes (overlap " << segment_overlap_seconds << "s)" << std::endl;

    const auto tmp_dir = CreateTempDir();
    std::cout << "[Info] Temp dir: " << tmp_dir.string() << std::endl;

    int stems = -1;
    std::vector<drwav> writers;
    std::vector<std::string> stem_paths;
    std::vector<std::vector<float>> prev_tails;

    drwav_data_format format{};
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = 2;
    format.sampleRate = in_sr;
    format.bitsPerSample = 32;

    auto cleanup_tmp = [&]() {
        if (keep_temps) return;
        std::error_code ec;
        std::filesystem::remove_all(tmp_dir, ec);
    };

    try {
        for (size_t i = 0; i < seg_count; ++i) {
            const drwav_uint64 nominal_start = static_cast<drwav_uint64>(i) * segment_frames;
            const drwav_uint64 nominal_end = std::min(total_frames, nominal_start + segment_frames);
            const drwav_uint64 seg_start = (i == 0 || overlap_frames == 0) ? nominal_start
                                                                           : (nominal_start > overlap_frames ? (nominal_start - overlap_frames) : 0);
            const drwav_uint64 seg_frames = (nominal_end > seg_start) ? (nominal_end - seg_start) : 0;

            std::ostringstream seg_name;
            seg_name << "segment_" << std::setw(6) << std::setfill('0') << i << ".wav";
            const std::filesystem::path seg_out_base = tmp_dir / seg_name.str();

            std::vector<std::string> child_args;
            child_args.reserve(32);
            child_args.push_back(model_path);
            child_args.push_back(input_path);
            child_args.push_back(seg_out_base.string());

            if (chunk_size_set) {
                child_args.push_back("--chunk-size");
                child_args.push_back(std::to_string(chunk_size));
            }
            if (num_overlap_set) {
                child_args.push_back("--overlap");
                child_args.push_back(std::to_string(num_overlap));
            }
            if (!use_io_threads) {
                child_args.push_back("--no-io-threads");
            }
            if (!use_pipelined_stream) {
                child_args.push_back("--no-pipeline");
            }

            child_args.push_back("--start-frame");
            child_args.push_back(std::to_string(seg_start));
            child_args.push_back("--frames");
            child_args.push_back(std::to_string(seg_frames));
            child_args.push_back("--no-progress");

            std::cout << "[Info] Running segment " << (i + 1) << "/" << seg_count
                      << " (frames " << seg_start << " .. " << nominal_end
                      << ", len=" << seg_frames << ")" << std::endl;

            int rc = cli_process::SpawnChildAndWait(exe_path, child_args);
            if (rc != 0) {
                throw std::runtime_error("Child process failed for segment " + std::to_string(i) +
                                         " (exit code " + std::to_string(rc) + ")");
            }

            if (i == 0) {
                stems = DetectNumStemsFromSegmentOutputs(seg_out_base);
                std::cout << "[Info] Detected " << stems << " stems from model output." << std::endl;

                writers.resize(static_cast<size_t>(stems));
                stem_paths.resize(static_cast<size_t>(stems));
                prev_tails.resize(static_cast<size_t>(stems));

                for (int s = 0; s < stems; ++s) {
                    stem_paths[static_cast<size_t>(s)] = MakeStemOutputPath(output_path, s, stems);
                    if (!drwav_init_file_write(&writers[static_cast<size_t>(s)], stem_paths[static_cast<size_t>(s)].c_str(), &format, nullptr)) {
                        for (int j = 0; j < s; ++j) drwav_uninit(&writers[static_cast<size_t>(j)]);
                        throw std::runtime_error("Failed to open file for writing: " + stem_paths[static_cast<size_t>(s)]);
                    }
                }
            }

            for (int s = 0; s < stems; ++s) {
                const std::filesystem::path seg_stem = MakeStemOutputPath(seg_out_base.string(), s, stems);
                MergeOneSegmentStem(seg_stem, writers[static_cast<size_t>(s)], i, seg_count, overlap_frames, prev_tails[static_cast<size_t>(s)]);
            }

            if (!keep_temps) {
                std::error_code ec;
                if (stems <= 1) {
                    std::filesystem::remove(seg_out_base, ec);
                } else {
                    for (int s = 0; s < stems; ++s) {
                        const std::filesystem::path seg_stem = MakeStemOutputPath(seg_out_base.string(), s, stems);
                        std::filesystem::remove(seg_stem, ec);
                    }
                }
            }
        }

        for (int s = 0; s < stems; ++s) {
            drwav_uninit(&writers[static_cast<size_t>(s)]);
            std::cout << "Saved output stem " << s << ": " << stem_paths[static_cast<size_t>(s)] << std::endl;
        }

        cleanup_tmp();
        return 0;
    } catch (...) {
        for (auto& w : writers) {
            drwav_uninit(&w);
        }
        cleanup_tmp();
        throw;
    }
}

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " <model.gguf> <input.wav> <output.wav> [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --chunk-size <N>   Chunk size in samples (default: from model, fallback 352800)" << std::endl;
    std::cerr << "  --overlap <N>      Number of overlaps for crossfade (default: from model, fallback 2)" << std::endl;
    std::cerr << "  --no-stream        Disable streaming I/O (debug only; uses more RAM)" << std::endl;
    std::cerr << "  --no-io-threads    Streaming I/O without reader/writer threads (debug only)" << std::endl;
    std::cerr << "  --no-pipeline      Disable pipelined streaming inference (debug only)" << std::endl;
    std::cerr << "  --segment-minutes [N] Enable multiprocess segmentation (default N=30; auto-enabled for >30min)" << std::endl;
    std::cerr << "  --segment-overlap-seconds <N> Overlap duration for segment crossfade (default: 10)" << std::endl;
    std::cerr << "  --segment-keep-temp Keep temporary segment outputs (debug only)" << std::endl;
    std::cerr << "  --no-segment       Disable multiprocess segmentation (debug only)" << std::endl;
    std::cerr << "  --start-frame <N>  (Advanced) Start at PCM frame N (0-based)" << std::endl;
    std::cerr << "  --frames <N>       (Advanced) Process only N PCM frames from start-frame" << std::endl;
    std::cerr << "  --no-progress      Disable progress bar output" << std::endl;
    std::cerr << "  --pipeline-depth <N>  Streaming pipeline depth (1-8, default: 2)" << std::endl;
    std::cerr << "  --cuda-pinned-staging Enable CUDA pinned staging (default: off)" << std::endl;
    std::cerr << "  --help, -h         Show this help message" << std::endl;
    std::cerr << "  --version, -v      Show version information" << std::endl;
}

int main(int argc, char* argv[]) {
    // Default values (will be overridden by model defaults if not explicitly set)
    int chunk_size = -1;  // -1 means use model default
    int num_overlap = -1; // -1 means use model default
    bool chunk_size_set = false;
    bool num_overlap_set = false;
    bool use_streaming_io = true; // Default: streaming
    bool use_pipelined_stream = true; // Default: pipelined streaming inference
    bool use_io_threads = true; // Default: threaded reader/writer for streaming I/O
    enum class SegmentMode { Auto, On, Off };
    SegmentMode segment_mode = SegmentMode::Auto; // Default: auto-enable segmentation for long audio
    int segment_minutes = 30;
    int segment_overlap_seconds = 10;
    bool segment_keep_temp = false;
    bool no_progress = false;
    drwav_uint64 start_frame = 0;
    drwav_uint64 frames_limit = 0;
    bool frames_limit_set = false;
    
    // Check for help/version flags first
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--version" || arg == "-v") {
            std::cout << "bs-roformer-cli " << BSR_VERSION << std::endl;
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
    const auto exe_path = cli_process::GetSelfExecutablePath(argv[0]);
    
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
        } else if (arg == "--no-io-threads") {
            use_io_threads = false;
        } else if (arg == "--no-pipeline") {
            use_pipelined_stream = false;
        } else if (arg == "--segment-minutes") {
            segment_mode = SegmentMode::On;
            segment_minutes = 30;
            if (i + 1 < argc) {
                std::string next = argv[i + 1];
                if (!next.empty() && next[0] != '-') {
                    try {
                        segment_minutes = std::stoi(argv[++i]);
                    } catch (...) {
                        std::cerr << "Error: invalid segment-minutes" << std::endl;
                        return 1;
                    }
                }
            }
        } else if (arg == "--no-segment") {
            segment_mode = SegmentMode::Off;
        } else if (arg == "--segment-overlap-seconds" && i + 1 < argc) {
            try {
                segment_overlap_seconds = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Error: invalid segment-overlap-seconds" << std::endl;
                return 1;
            }
        } else if (arg == "--segment-keep-temp") {
            segment_keep_temp = true;
        } else if (arg == "--start-frame" && i + 1 < argc) {
            try {
                start_frame = static_cast<drwav_uint64>(std::stoull(argv[++i]));
            } catch (...) {
                std::cerr << "Error: invalid start-frame" << std::endl;
                return 1;
            }
        } else if (arg == "--frames" && i + 1 < argc) {
            try {
                frames_limit = static_cast<drwav_uint64>(std::stoull(argv[++i]));
                frames_limit_set = true;
            } catch (...) {
                std::cerr << "Error: invalid frames" << std::endl;
                return 1;
            }
        } else if (arg == "--no-progress") {
            no_progress = true;
        } else if (arg == "--pipeline-depth" && i + 1 < argc) {
            try {
                int depth = std::stoi(argv[++i]);
                if (depth < 1 || depth > 8) {
                    std::cerr << "Error: pipeline-depth must be 1-8" << std::endl;
                    return 1;
                }
                BSRConfig::SetPipelineDepth(depth);
            } catch (...) {
                std::cerr << "Error: invalid pipeline-depth" << std::endl;
                return 1;
            }
        } else if (arg == "--cuda-pinned-staging") {
            BSRConfig::SetCudaPinnedStaging(true);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        if (!use_streaming_io) {
            if (segment_mode == SegmentMode::On) {
                throw std::runtime_error("Multiprocess segmentation requires streaming I/O (do not use --no-stream)");
            }
            // Segmentation requires streaming I/O. Keep behaviour predictable if user requests --no-stream.
            segment_mode = SegmentMode::Off;
        }
        if (frames_limit_set || start_frame != 0) {
            if (segment_mode == SegmentMode::On) {
                throw std::runtime_error("Do not combine --segment-minutes with --start-frame/--frames");
            }
            // Segmentation doesn't support partial range processing; disable automatically.
            segment_mode = SegmentMode::Off;
        }

        bool use_segmentation = false;
        if (segment_mode != SegmentMode::Off) {
            if (segment_minutes <= 0) {
                throw std::runtime_error("segment-minutes must be a positive integer");
            }

            if (segment_mode == SegmentMode::On) {
                use_segmentation = true;
            } else {
                // Auto: enable only when the input is longer than one segment.
                drwav in_wav{};
                if (!drwav_init_file(&in_wav, input_path.c_str(), nullptr)) {
                    throw std::runtime_error("Failed to open audio file: " + input_path);
                }
                const unsigned int in_sr = in_wav.sampleRate;
                const drwav_uint64 total_frames = in_wav.totalPCMFrameCount;
                drwav_uninit(&in_wav);

                const drwav_uint64 segment_frames = static_cast<drwav_uint64>(segment_minutes) * 60ull *
                                                    static_cast<drwav_uint64>(in_sr);
                if (segment_frames > 0 && total_frames > segment_frames) {
                    use_segmentation = true;
                }
            }
        }

        if (use_segmentation) {
            if (!use_streaming_io) {
                throw std::runtime_error("Multiprocess segmentation requires streaming I/O (do not use --no-stream)");
            }

            return RunSegmentedMultiprocess(exe_path,
                                           model_path,
                                           input_path,
                                           output_path,
                                           chunk_size_set,
                                           chunk_size,
                                           num_overlap_set,
                                           num_overlap,
                                           use_pipelined_stream,
                                           use_io_threads,
                                           segment_minutes,
                                           segment_overlap_seconds,
                                           segment_keep_temp);
        }

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
        auto progress_callback = [&](float progress) {
            if (no_progress) return;
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
            if (frames_limit_set || start_frame != 0) {
                throw std::runtime_error("--start-frame/--frames requires streaming mode (do not use --no-stream)");
            }
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

            if (start_frame > total_frames) {
                drwav_uninit(&wav);
                throw std::runtime_error("start-frame is beyond end of file");
            }
            if (start_frame > 0) {
                if (!drwav_seek_to_pcm_frame(&wav, start_frame)) {
                    drwav_uninit(&wav);
                    throw std::runtime_error("Failed to seek to start-frame");
                }
            }
            const drwav_uint64 available_frames = total_frames - start_frame;
            const drwav_uint64 total_frames_to_process = frames_limit_set ? std::min(frames_limit, available_frames) : available_frames;

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
            
            if (use_io_threads) {
                // Keep I/O queues/pools small to reduce peak RSS on long runs.
                // (Reader/writer threads are fast; deep buffering is usually unnecessary.)
                const size_t io_queue_depth = 2;

                ThreadSafeQueue<InputChunk> input_queue(io_queue_depth);
                ThreadSafeQueue<OutputChunk> output_queue(io_queue_depth);
                std::atomic<drwav_uint64> frames_read_total{0};

                // Pool large buffers to avoid per-chunk heap growth (Windows heap commit can climb with repeated alloc/free).
                ThreadSafeQueue<std::vector<float>> input_pool(io_queue_depth);
                ThreadSafeQueue<std::vector<std::vector<float>>> output_pool(io_queue_depth);
                for (size_t i = 0; i < io_queue_depth; ++i) {
                    input_pool.push(std::vector<float>{});

                    std::vector<std::vector<float>> out_buf;
                    out_buf.resize(static_cast<size_t>(stems));
                    for (int s = 0; s < stems; ++s) {
                        out_buf[static_cast<size_t>(s)].reserve(static_cast<size_t>(read_frames) * 2);
                    }
                    output_pool.push(std::move(out_buf));
                }

                // Reader thread
                std::thread reader([&]() {
                    std::vector<float> read_buf(static_cast<size_t>(read_frames) * in_ch);
                    drwav_uint64 frames_left = total_frames_to_process;
                    while (true) {
                        if (frames_left == 0) break;
                        const drwav_uint64 want = std::min(read_frames, frames_left);
                        drwav_uint64 got = drwav_read_pcm_frames_f32(&wav, want, read_buf.data());
                        if (got == 0) break;
                        frames_left -= got;

                        std::vector<float> chunk_buf;
                        input_pool.pop(chunk_buf);

                        InputChunk chunk;
                        if (in_ch == 1) {
                            chunk_buf.resize(static_cast<size_t>(got) * 2);
                            for (drwav_uint64 i = 0; i < got; ++i) {
                                float x = read_buf[static_cast<size_t>(i)];
                                chunk_buf[static_cast<size_t>(i) * 2 + 0] = x;
                                chunk_buf[static_cast<size_t>(i) * 2 + 1] = x;
                            }
                        } else {
                            const size_t n_floats = static_cast<size_t>(got) * in_ch;
                            chunk_buf.resize(n_floats);
                            std::memcpy(chunk_buf.data(), read_buf.data(), n_floats * sizeof(float));
                        }
                        chunk.data = std::move(chunk_buf);
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

                        output_pool.push(std::move(chunk.stems));
                    }
                });

                // Main processing loop
                float last_progress = -0.05f;
                while (true) {
                    InputChunk chunk;
                    input_queue.pop(chunk);
                    if (chunk.is_end) break;

                    std::vector<std::vector<float>> out_buf;
                    output_pool.pop(out_buf);

                    engine.ProcessStreamInto(*stream, chunk.data, out_buf);
                    input_pool.push(std::move(chunk.data));

                    OutputChunk out_chunk;
                    out_chunk.stems = std::move(out_buf);
                    output_queue.push(std::move(out_chunk));

                    if (total_frames_to_process > 0) {
                        float progress = static_cast<float>(frames_read_total.load()) / static_cast<float>(total_frames_to_process);
                        if (progress - last_progress >= 0.05f) {
                            progress_callback(progress);
                            last_progress = progress;
                        }
                    }
                }

                std::vector<std::vector<float>> tail;
                output_pool.pop(tail);
                engine.FinalizeStreamInto(*stream, tail);

                bool has_tail = false;
                for (const auto& s : tail) {
                    if (!s.empty()) {
                        has_tail = true;
                        break;
                    }
                }

                if (has_tail) {
                    OutputChunk tail_chunk;
                    tail_chunk.stems = std::move(tail);
                    output_queue.push(std::move(tail_chunk));
                } else {
                    output_pool.push(std::move(tail));
                }

                OutputChunk end; end.is_end = true;
                output_queue.push(std::move(end));

                reader.join();
                writer.join();
            } else {
                std::cout << "[Info] Streaming I/O without reader/writer threads (--no-io-threads)" << std::endl;

                std::vector<float> read_buf(static_cast<size_t>(read_frames) * in_ch);
                std::vector<float> chunk_data;
                std::vector<std::vector<float>> out;
                drwav_uint64 frames_read_total = 0;
                drwav_uint64 frames_left = total_frames_to_process;
                float last_progress = -0.05f;

                while (true) {
                    if (frames_left == 0) break;
                    const drwav_uint64 want = std::min(read_frames, frames_left);
                    drwav_uint64 got = drwav_read_pcm_frames_f32(&wav, want, read_buf.data());
                    if (got == 0) break;
                    frames_left -= got;

                    if (in_ch == 1) {
                        chunk_data.resize(static_cast<size_t>(got) * 2);
                        for (drwav_uint64 i = 0; i < got; ++i) {
                            float x = read_buf[static_cast<size_t>(i)];
                            chunk_data[static_cast<size_t>(i) * 2 + 0] = x;
                            chunk_data[static_cast<size_t>(i) * 2 + 1] = x;
                        }
                    } else {
                        chunk_data.assign(read_buf.begin(), read_buf.begin() + static_cast<size_t>(got) * in_ch);
                    }
                    frames_read_total += got;

                    engine.ProcessStreamInto(*stream, chunk_data, out);
                    for (int s = 0; s < stems; ++s) {
                        if (out.size() <= static_cast<size_t>(s)) continue;
                        drwav_write_pcm_frames(&writers[s], out[s].size() / 2, out[s].data());
                    }

                    if (total_frames_to_process > 0) {
                        float progress = static_cast<float>(frames_read_total) / static_cast<float>(total_frames_to_process);
                        if (progress - last_progress >= 0.05f) {
                            progress_callback(progress);
                            last_progress = progress;
                        }
                    }
                }

                std::vector<std::vector<float>> tail;
                engine.FinalizeStreamInto(*stream, tail);
                for (int s = 0; s < stems; ++s) {
                    if (tail.size() <= static_cast<size_t>(s)) continue;
                    if (!tail[s].empty()) {
                        drwav_write_pcm_frames(&writers[s], tail[s].size() / 2, tail[s].data());
                    }
                }
            }

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
