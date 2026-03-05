#include "test_common.h"
#include "bs_roformer/inference.h"
#include <random>
#include <algorithm>
#include <deque>

static std::vector<float> MakeRandomInterleavedStereo(int n_samples_per_ch, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> out(n_samples_per_ch * 2);
    for (int i = 0; i < n_samples_per_ch; ++i) {
        out[i * 2 + 0] = dist(rng);
        out[i * 2 + 1] = dist(rng);
    }
    return out;
}

static void AppendStems(std::vector<std::vector<float>>& dst, const std::vector<std::vector<float>>& src) {
    if (src.empty()) return;
    if (dst.empty()) dst.resize(src.size());
    for (size_t s = 0; s < src.size(); ++s) {
        dst[s].insert(dst[s].end(), src[s].begin(), src[s].end());
    }
}

int main() {
    std::cout << "Test: Streaming Overlap-Add matches ProcessOverlapAdd (identity model)" << std::endl;

    // Keep this small for fast unit test + easier debugging.
    // Must be divisible by 10 because fade_size = chunk_size / 10.
    const int chunk_size = 1000;
    const int num_overlap = 2;

    auto identity = [](const std::vector<float>& chunk) {
        return std::vector<std::vector<float>>{chunk};
    };

    // Cover both do_pad=false (<= 2*border) and do_pad=true (> 2*border)
    const int test_lengths[] = {200, 800, 1000, 1001, 1200, 4096};
    const int push_sizes[] = {7, 123, 499, 777}; // irregular push sizes to stress buffering

    for (int n_samples : test_lengths) {
        auto input = MakeRandomInterleavedStereo(n_samples, static_cast<uint32_t>(n_samples));
        auto expected = Inference::ProcessOverlapAdd(input, chunk_size, num_overlap, identity);

        Inference::OverlapAddStreamer streamer(chunk_size, num_overlap, /*num_stems=*/1, identity);

        std::vector<std::vector<float>> actual;
        int cursor = 0;
        int push_idx = 0;
        while (cursor < n_samples) {
            int push = push_sizes[push_idx++ % (sizeof(push_sizes) / sizeof(push_sizes[0]))];
            int take = std::min(push, n_samples - cursor);

            // take samples per channel => interleaved float count is take*2
            std::vector<float> slice(input.begin() + cursor * 2, input.begin() + (cursor + take) * 2);
            auto out = streamer.Push(slice);
            AppendStems(actual, out);

            cursor += take;
        }

        auto tail = streamer.Finalize();
        AppendStems(actual, tail);

        // Also test the "manual schedule/consume" API which enables pipelined streaming.
        // This simulates a fixed in-flight depth and delayed consumption.
        Inference::OverlapAddStreamer manual(chunk_size, num_overlap, /*num_stems=*/1, nullptr);
        std::deque<Inference::OverlapAddStreamer::ScheduledChunk> inflight;
        const size_t depth = 3;

        std::vector<std::vector<float>> actual_manual;
        cursor = 0;
        push_idx = 0;
        while (cursor < n_samples) {
            int push = push_sizes[push_idx++ % (sizeof(push_sizes) / sizeof(push_sizes[0]))];
            int take = std::min(push, n_samples - cursor);

            std::vector<float> slice(input.begin() + cursor * 2, input.begin() + (cursor + take) * 2);
            manual.Feed(slice);
            cursor += take;

            Inference::OverlapAddStreamer::ScheduledChunk scheduled;
            while (inflight.size() < depth && manual.TryScheduleNext(scheduled)) {
                inflight.push_back(std::move(scheduled));
            }

            if (inflight.size() == depth) {
                auto chunk = std::move(inflight.front());
                inflight.pop_front();

                auto out = manual.ConsumeScheduled(chunk, identity(chunk.chunk_in));
                AppendStems(actual_manual, out);
            }
        }

        manual.FinalizeInput();
        for (;;) {
            Inference::OverlapAddStreamer::ScheduledChunk scheduled;
            while (inflight.size() < depth && manual.TryScheduleNext(scheduled)) {
                inflight.push_back(std::move(scheduled));
            }

            if (inflight.empty()) break;

            auto chunk = std::move(inflight.front());
            inflight.pop_front();

            auto out = manual.ConsumeScheduled(chunk, identity(chunk.chunk_in));
            AppendStems(actual_manual, out);
        }

        bool pass = CompareAndReport("stream-vs-batch",
                                     expected[0].data(),
                                     expected[0].size(),
                                     actual[0].data(),
                                     actual[0].size(),
                                     1e-6f,
                                     1e-6f);
        if (!pass) {
            LOG_FAIL();
            return 1;
        }

        pass = CompareAndReport("manual-schedule-vs-batch",
                                expected[0].data(),
                                expected[0].size(),
                                actual_manual[0].data(),
                                actual_manual[0].size(),
                                1e-6f,
                                1e-6f);
        if (!pass) {
            LOG_FAIL();
            return 1;
        }
    }

    LOG_PASS();
    return 0;
}
