#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <ggml.h>

// Tensor comparison result
struct TensorComparison {
    std::string name;
    bool match;
    float max_abs_diff;
    float mean_abs_diff;
    float max_rel_diff;
    std::vector<size_t> shape_expected;
    std::vector<size_t> shape_actual;
};

// Utility functions
namespace utils {

/**
 * Load numpy .npy file into memory
 * Returns: pointer to data (caller must free), shape vector
 */
std::pair<float*, std::vector<size_t>> load_npy(const std::string& filepath);

/**
 * Load all weights from debug_tensors/weights/ directory
 * Returns: map of tensor name -> (data pointer, shape)
 */
std::map<std::string, std::pair<float*, std::vector<size_t>>> load_all_weights(const std::string& debug_dir);

/**
 * Load activation tensor from debug_tensors/activations/
 */
std::pair<float*, std::vector<size_t>> load_activation(const std::string& debug_dir, const std::string& name);

/**
 * Compare two tensors (expected from numpy, actual from ggml)
 * Returns comparison result with detailed statistics
 */
TensorComparison compare_tensors(
    const std::string& name,
    const float* expected,
    const std::vector<size_t>& expected_shape,
    const ggml_tensor* actual,
    float atol = 1e-4,
    float rtol = 1e-3
);

/**
 * Print comparison result
 */
void print_comparison(const TensorComparison& cmp, bool verbose = false);

/**
 * Create ggml tensor from numpy data
 */
ggml_tensor* create_tensor_from_numpy(
    ggml_context* ctx,
    const float* data,
    const std::vector<size_t>& shape
);

/**
 * Get total number of elements in shape
 */
size_t shape_nelements(const std::vector<size_t>& shape);

/**
 * Print tensor shape for debugging
 */
void print_tensor_shape(const std::string& name, const ggml_tensor* tensor);

/**
 * Free numpy data pointer
 */
void free_npy_data(float* data);

} // namespace utils
