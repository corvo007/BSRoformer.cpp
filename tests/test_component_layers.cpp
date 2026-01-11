#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cstring>
#include <cstdlib>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include "../src/model.h"
#include "../src/utils.h"

/**
 * test_component_layers.cpp
 * 
 * Verifies Transformer layers against golden tensors from export_debug.py
 * Copied from tests_old/test_component_layers.cpp with env var support
 */

std::string GetModelPath() {
    const char* env = std::getenv("MBR_MODEL_PATH");
    return env ? env : "mel_band_roformer.gguf";
}

std::string GetTestDataDir() {
    const char* env = std::getenv("MBR_TEST_DATA_DIR");
    return env ? env : ".";
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Test: Transformer Layers Verification" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::string model_path = GetModelPath();
    std::string debug_dir = GetTestDataDir();
    
    if (argc > 1) model_path = argv[1];
    if (argc > 2) debug_dir = argv[2];
    
    try {
        // 1. Load Model
        std::cout << "\n[1/6] Loading model..." << std::endl;
        MelBandRoformer model;
        model.Initialize(model_path);
        
        // 2. Load golden tensors
        std::cout << "\n[2/6] Loading golden tensors..." << std::endl;
        
        // Load after_band_split (input to Transformers)
        auto [input_data, input_shape] = utils::load_activation(debug_dir, "after_band_split");
        if (!input_data) {
            std::cerr << "Failed to load after_band_split.npy" << std::endl;
            return 1;
        }
        std::cout << "  Input (after_band_split) shape: [";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            std::cout << input_shape[i];
            if (i < input_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Load before_mask_est (expected output after all 6 layers)
        auto [expected_data, expected_shape] = utils::load_activation(debug_dir, "before_mask_est");
        if (!expected_data) {
            std::cerr << "Failed to load before_mask_est.npy" << std::endl;
            utils::free_npy_data(input_data);
            return 1;
        }
        std::cout << "  Expected (before_mask_est) shape: [";
        for (size_t i = 0; i < expected_shape.size(); ++i) {
            std::cout << expected_shape[i];
            if (i < expected_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Extract dimensions from shapes
        // PyTorch: [batch, time, bands, dim]
        int batch = static_cast<int>(input_shape[0]);
        int n_frames = static_cast<int>(input_shape[1]);
        int n_bands = static_cast<int>(input_shape[2]);
        int dim = static_cast<int>(input_shape[3]);
        
        std::cout << "  batch=" << batch << ", n_frames=" << n_frames 
                  << ", n_bands=" << n_bands << ", dim=" << dim << std::endl;
        
        // 3. Build computation graph
        std::cout << "\n[3/6] Building computation graph..." << std::endl;
        
        size_t mem_size = 1024 * 1024 * 1024;  // 1GB for Transformers
        struct ggml_init_params ctx_params = {
            /*.mem_size   = */ mem_size,
            /*.mem_buffer = */ nullptr,
            /*.no_alloc   = */ true,
        };
        ggml_context* ctx = ggml_init(ctx_params);
        
        // Expanded position tensors for CUDA RoPE compatibility:
        // pos_time_exp: size [T * F * B], repeating [0..T-1] for each F*B batch
        // pos_freq_exp: size [F * T * B], repeating [0..F-1] for each T*B batch
        int time_exp_size = n_frames * n_bands * batch;  // T * F * B
        int freq_exp_size = n_bands * n_frames * batch;  // F * T * B
        
        std::vector<int32_t> pos_time_exp_data(time_exp_size);
        for (int i = 0; i < time_exp_size; ++i) {
            pos_time_exp_data[i] = i % n_frames;  // Repeat [0..T-1]
        }
        
        std::vector<int32_t> pos_freq_exp_data(freq_exp_size);
        for (int i = 0; i < freq_exp_size; ++i) {
            pos_freq_exp_data[i] = i % n_bands;  // Repeat [0..F-1]
        }
        
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, false);
        
        // Create input tensor: [dim, bands, time, batch] (GGML order)
        ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                                 dim, n_bands, n_frames, batch);
        ggml_set_input(input);
        
        // Create expanded position tensors for RoPE
        ggml_tensor* pos_time_exp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, time_exp_size);
        ggml_set_input(pos_time_exp);
        
        ggml_tensor* pos_freq_exp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, freq_exp_size);
        ggml_set_input(pos_freq_exp);
        
        // Build Transformers graph
        ggml_tensor* x = model.BuildTransformersGraph(ctx, input, gf, pos_time_exp, pos_freq_exp, n_frames, batch);
        if (!x) {
            std::cerr << "FAILED: BuildTransformersGraph returned nullptr" << std::endl;
            utils::free_npy_data(input_data);
            utils::free_npy_data(expected_data);
            ggml_free(ctx);
            return 1;
        }
        
        // Mark output
        ggml_tensor* output = ggml_dup(ctx, x);
        ggml_set_output(output);
        ggml_build_forward_expand(gf, output);
        
        std::cout << "  Graph built with " << ggml_graph_n_nodes(gf) << " nodes" << std::endl;
        
        // 4. Allocate and execute
        std::cout << "\n[4/6] Allocating graph..." << std::endl;
        
        ggml_gallocr_t allocr = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(model.GetBackend())
        );
        
        if (!ggml_gallocr_alloc_graph(allocr, gf)) {
            std::cerr << "FAILED: Failed to allocate graph" << std::endl;
            utils::free_npy_data(input_data);
            utils::free_npy_data(expected_data);
            ggml_gallocr_free(allocr);
            ggml_free(ctx);
            return 1;
        }
        
        std::cout << "\n[5/6] Executing graph..." << std::endl;
        
        // Copy input data
        ggml_backend_tensor_set(input, input_data, 0, ggml_nbytes(input));
        
        // Copy expanded position tensors
        ggml_backend_tensor_set(pos_time_exp, pos_time_exp_data.data(), 0, ggml_nbytes(pos_time_exp));
        ggml_backend_tensor_set(pos_freq_exp, pos_freq_exp_data.data(), 0, ggml_nbytes(pos_freq_exp));
        
        // Compute
        ggml_backend_graph_compute(model.GetBackend(), gf);
        
        // 5. Compare results
        std::cout << "\n[6/6] Comparing results..." << std::endl;
        
        // Copy output from GPU to CPU for comparison
        std::vector<float> output_data(ggml_nelements(output));
        ggml_backend_tensor_get(output, output_data.data(), 0, ggml_nbytes(output));
        
        // Compare element counts
        size_t expected_nelements = utils::shape_nelements(expected_shape);
        std::cout << "  Output elements: " << output_data.size() << std::endl;
        std::cout << "  Expected elements: " << expected_nelements << std::endl;
        
        // Compute comparison statistics directly
        float max_abs = 0.0f;
        float sum_abs = 0.0f;
        for (size_t i = 0; i < output_data.size() && i < expected_nelements; ++i) {
            float diff = std::abs(expected_data[i] - output_data[i]);
            max_abs = std::max(max_abs, diff);
            sum_abs += diff;
        }
        float mean_abs = sum_abs / output_data.size();
        
        std::cout << "\n[Comparison] Transformers Output" << std::endl;
        std::cout << "  Max abs diff:  " << max_abs << std::endl;
        std::cout << "  Mean abs diff: " << mean_abs << std::endl;
        
        bool match = max_abs <= 3e-2f || mean_abs <= 3e-3f;
        
        // Cleanup
        utils::free_npy_data(input_data);
        utils::free_npy_data(expected_data);
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        
        if (match) {
            std::cout << "\nPASSED: Transformers match PyTorch output" << std::endl;
            return 0;
        } else {
            std::cout << "\nFAILED: Transformers do not match PyTorch output" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
