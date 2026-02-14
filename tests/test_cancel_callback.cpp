#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "bs_roformer/inference.h"

static bool IsCancelledError(const std::exception& e) {
    return std::string(e.what()) == "Inference cancelled";
}

int main() {
    std::cout << "Test: Cancel Callback Behavior" << std::endl;

    const int channels = 2;
    const int samples = 96;
    const int chunk_size = 32;
    const int num_overlap = 2;

    std::vector<float> input(samples * channels);
    for (int i = 0; i < samples; ++i) {
        input[i * channels + 0] = std::sin(0.1f * static_cast<float>(i));
        input[i * channels + 1] = std::cos(0.1f * static_cast<float>(i));
    }

    auto identity = [](const std::vector<float>& chunk) {
        return std::vector<std::vector<float>>{chunk};
    };

    // Case 1: immediate cancellation
    bool immediate_cancelled = false;
    try {
        (void)Inference::ProcessOverlapAdd(
            input,
            chunk_size,
            num_overlap,
            identity,
            nullptr,
            []() { return true; });
    } catch (const std::exception& e) {
        if (!IsCancelledError(e)) {
            std::cerr << "Unexpected exception for immediate cancel: " << e.what() << std::endl;
            return 1;
        }
        immediate_cancelled = true;
    }

    if (!immediate_cancelled) {
        std::cerr << "Immediate cancellation did not throw" << std::endl;
        return 1;
    }

    // Case 2: delayed cancellation
    int cancel_calls = 0;
    bool delayed_cancelled = false;
    try {
        (void)Inference::ProcessOverlapAdd(
            input,
            chunk_size,
            num_overlap,
            identity,
            nullptr,
            [&cancel_calls]() {
                ++cancel_calls;
                return cancel_calls >= 3;
            });
    } catch (const std::exception& e) {
        if (!IsCancelledError(e)) {
            std::cerr << "Unexpected exception for delayed cancel: " << e.what() << std::endl;
            return 1;
        }
        delayed_cancelled = true;
    }

    if (!delayed_cancelled) {
        std::cerr << "Delayed cancellation did not throw" << std::endl;
        return 1;
    }

    // Case 3: cancel callback always false should match baseline output.
    auto no_cancel = []() { return false; };
    auto baseline = Inference::ProcessOverlapAdd(input, chunk_size, num_overlap, identity);
    auto with_no_cancel = Inference::ProcessOverlapAdd(
        input,
        chunk_size,
        num_overlap,
        identity,
        nullptr,
        no_cancel);

    if (baseline.size() != with_no_cancel.size() || baseline.empty()) {
        std::cerr << "Output stem count mismatch in no-cancel path" << std::endl;
        return 1;
    }

    if (baseline[0].size() != with_no_cancel[0].size()) {
        std::cerr << "Output sample count mismatch in no-cancel path" << std::endl;
        return 1;
    }

    float max_diff = 0.0f;
    for (size_t i = 0; i < baseline[0].size(); ++i) {
        max_diff = std::max(max_diff, std::abs(baseline[0][i] - with_no_cancel[0][i]));
    }

    if (max_diff > 1e-6f) {
        std::cerr << "No-cancel output mismatch, max diff = " << max_diff << std::endl;
        return 1;
    }

    std::cout << "PASSED" << std::endl;
    return 0;
}
