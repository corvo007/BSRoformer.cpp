#include "utils.h"
#include <ggml-backend.h>
#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

namespace utils {

// NPY header parsing (simplified - assumes float32, C-order)
std::pair<float*, std::vector<size_t>> load_npy(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open: " << filepath << std::endl;
        return {nullptr, {}};
    }
    
    // Read magic string
    char magic[6];
    file.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") {
        std::cerr << "Invalid NPY file: " << filepath << std::endl;
        return {nullptr, {}};
    }
    
    // Read version
    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    
    // Read header length
    uint16_t header_len;
    if (major == 1) {
        file.read(reinterpret_cast<char*>(&header_len), 2);
    } else {
        uint32_t header_len_32;
        file.read(reinterpret_cast<char*>(&header_len_32), 4);
        header_len = header_len_32;
    }
    
    // Read header
    std::string header(header_len, ' ');
    file.read(&header[0], header_len);
    
    // Parse shape from header
    std::vector<size_t> shape;
    size_t shape_start = header.find("'shape': (");
    if (shape_start == std::string::npos) {
        shape_start = header.find("\"shape\": (");
    }
    
    if (shape_start != std::string::npos) {
        size_t shape_end = header.find(')', shape_start);
        std::string shape_str = header.substr(shape_start + 10, shape_end - shape_start - 10);
        std::istringstream ss(shape_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            // Remove spaces
            token.erase(std::remove(token.begin(), token.end(), ' '), token.end());
            if (!token.empty()) {
                shape.push_back(std::stoull(token));
            }
        }
    }
    
    // Calculate total elements
    size_t nelements = 1;
    for (size_t dim : shape) {
        nelements *= dim;
    }
    
    // Read data
    float* data = new float[nelements];
    file.read(reinterpret_cast<char*>(data), nelements * sizeof(float));
    
    file.close();
    return {data, shape};
}

std::map<std::string, std::pair<float*, std::vector<size_t>>> load_all_weights(const std::string& debug_dir) {
    std::map<std::string, std::pair<float*, std::vector<size_t>>> weights;
    std::string weights_dir = debug_dir + "/weights";
    
    for (const auto& entry : fs::directory_iterator(weights_dir)) {
        if (entry.path().extension() == ".npy") {
            std::string name = entry.path().stem().string();
            auto [data, shape] = load_npy(entry.path().string());
            if (data) {
                weights[name] = {data, shape};
                // std::cout << "Loaded weight: " << name << " shape: [";
                //for (size_t i = 0; i < shape.size(); ++i) {
                //    std::cout << shape[i];
                //    if (i < shape.size() - 1) std::cout << ", ";
                //}
                // std::cout << "]" << std::endl;
            }
        }
    }
    
    return weights;
}

std::pair<float*, std::vector<size_t>> load_activation(const std::string& debug_dir, const std::string& name) {
    std::string filepath = debug_dir + "/activations/" + name + ".npy";
    return load_npy(filepath);
}

TensorComparison compare_tensors(
    const std::string& name,
    const float* expected,
    const std::vector<size_t>& expected_shape,
    const ggml_tensor* actual,
    float atol,
    float rtol
) {
    TensorComparison result;
    result.name = name;
    result.shape_expected = expected_shape;
    
    // Extract actual shape
    std::vector<size_t> actual_shape;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (actual->ne[i] > 1 || i == 0) {
            actual_shape.push_back(actual->ne[i]);
        }
    }
    std::reverse(actual_shape.begin(), actual_shape.end());
    result.shape_actual = actual_shape;
    
    // Robust Squeeze Comparison
    std::vector<size_t> expected_squeezed;
    for (size_t dim : expected_shape) {
        if (dim > 1) expected_squeezed.push_back(dim);
    }
    
    std::vector<size_t> actual_squeezed;
    for (size_t dim : actual_shape) {
        if (dim > 1) actual_squeezed.push_back(dim);
    }
    
    bool shape_match = false;
    if (expected_squeezed.size() == actual_squeezed.size()) {
        shape_match = true;
        for (size_t i = 0; i < expected_squeezed.size(); ++i) {
            if (expected_squeezed[i] != actual_squeezed[i]) {
                shape_match = false;
                break;
            }
        }
    }

    if (!shape_match) {
        result.match = false;
        result.max_abs_diff = -1;
        result.mean_abs_diff = -1;
        result.max_rel_diff = -1;
        return result;
    }
    
    // Compare values
    size_t nelements = shape_nelements(expected_shape);
    // Note: shape_nelements uses full shape, which is correct as total elements match.
    
    // Safe data access for Backend/CUDA (Copy to CPU first)
    // Note: This requires including ggml-backend.h and linking against it
    std::vector<float> actual_data_vec(nelements);
    ggml_backend_tensor_get(const_cast<ggml_tensor*>(actual), actual_data_vec.data(), 0, ggml_nbytes(actual));
    const float* actual_data = actual_data_vec.data();
    
    float max_abs = 0.0f;
    float sum_abs = 0.0f;
    float max_rel = 0.0f;
    
    for (size_t i = 0; i < nelements; ++i) {
        float diff = std::abs(expected[i] - actual_data[i]);
        max_abs = std::max(max_abs, diff);
        sum_abs += diff;
        
        float rel_diff = 0.0f;
        if (std::abs(expected[i]) > 1e-8) {
            rel_diff = diff / std::abs(expected[i]);
        }
        max_rel = std::max(max_rel, rel_diff);
    }
    
    result.max_abs_diff = max_abs;
    result.mean_abs_diff = sum_abs / nelements;
    result.max_rel_diff = max_rel;
    result.match = (max_abs <= atol) || (max_rel <= rtol);
    
    return result;
}

void print_comparison(const TensorComparison& cmp, bool verbose) {
    std::cout << "\n[Comparison] " << cmp.name << std::endl;
    
    // Print shapes
    std::cout << "  Expected shape: [";
    for (size_t i = 0; i < cmp.shape_expected.size(); ++i) {
        std::cout << cmp.shape_expected[i];
        if (i < cmp.shape_expected.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "  Actual shape:   [";
    for (size_t i = 0; i < cmp.shape_actual.size(); ++i) {
        std::cout << cmp.shape_actual[i];
        if (i < cmp.shape_actual.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Print statistics
    if (cmp.max_abs_diff >= 0) {
        std::cout << "  Max abs diff:  " << cmp.max_abs_diff << std::endl;
        std::cout << "  Mean abs diff: " << cmp.mean_abs_diff << std::endl;
        std::cout << "  Max rel diff:  " << cmp.max_rel_diff << std::endl;
        std::cout << "  Status:        " << (cmp.match ? "✓ MATCH" : "✗ MISMATCH") << std::endl;
    } else {
        std::cout << "  Status:        ✗ SHAPE MISMATCH" << std::endl;
    }
}

ggml_tensor* create_tensor_from_numpy(
    ggml_context* ctx,
    const float* data,
    const std::vector<size_t>& shape
) {
    // GGML uses reversed dimension order
    int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    for (size_t i = 0; i < shape.size() && i < GGML_MAX_DIMS; ++i) {
        ne[shape.size() - 1 - i] = shape[i];
    }
    
    ggml_tensor* tensor = ggml_new_tensor(ctx, GGML_TYPE_F32, shape.size(), ne);
    memcpy(tensor->data, data, shape_nelements(shape) * sizeof(float));
    
    return tensor;
}

size_t shape_nelements(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (size_t dim : shape) {
        n *= dim;
    }
    return n;
}

void print_tensor_shape(const std::string& name, const ggml_tensor* tensor) {
    std::cout << name << " shape: [";
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        if (tensor->ne[i] > 1 || i == 0) {
            std::cout << tensor->ne[i];
            if (i > 0) std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

void free_npy_data(float* data) {
    delete[] data;
}

} // namespace utils
