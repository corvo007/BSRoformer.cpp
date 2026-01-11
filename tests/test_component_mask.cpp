#include "test_common.h"

int main(int argc, char* argv[]) {
    std::cout << "Test: MaskEstimator Component Verification" << std::endl;
    
    std::string model_path = GetModelPath();
    std::string data_dir = GetTestDataDir();
    
    if (argc > 1) model_path = argv[1];
    if (argc > 2) data_dir = argv[2];
    
    LOG_STEP(1, 4, "Loading model from " + model_path);
    MelBandRoformer model;
    model.Initialize(model_path);
    
    LOG_STEP(2, 4, "Loading golden tensors");
    GoldenTensor input(data_dir, "before_mask_est");
    GoldenTensor expected(data_dir, "mask_est0");
    
    TEST_ASSERT_LOAD(input, "before_mask_est");
    TEST_ASSERT_LOAD(expected, "mask_est0");
    
    input.PrintShape("Input");
    expected.PrintShape("Expected");
    
    // Input PyTorch: [1, T, Bands, Dim] -> [1, 301, 60, 64] ?
    // Let's check export_debug.py line 246
    // x (before_mask_est) comes from freq_transformer.
    // x shape is [batch, time, bands, dim] (rearranged in line 229: b t f d)
    // Wait, line 229 says: x = rearrange(x, 'b f t d -> b t f d')
    // So input is [B, T, Bands, Dim]
    
    int batch = input.shape[0];
    int n_frames = input.shape[1];
    int n_bands = input.shape[2];
    int dim = input.shape[3];
    
    // 3. Build Graph
    LOG_STEP(3, 4, "Building computation graph");
    TestContext tc(&model);
    
    // GGML Input: [Dim, Bands, Frames, Batch] (ne0=Dim)
    // Matches NumPy [B, T, Bands, Dim] layout directly
    ggml_tensor* in_tensor = ggml_new_tensor_4d(tc.ctx, GGML_TYPE_F32, dim, n_bands, n_frames, batch);
    ggml_set_input(in_tensor);
    
    ggml_tensor* out = model.BuildMaskEstimatorGraph(tc.ctx, in_tensor, tc.gf, n_frames, batch);
    TEST_ASSERT(out, "BuildMaskEstimatorGraph returned nullptr");
    
    ggml_build_forward_expand(tc.gf, out);
    
    // 4. Exec
    LOG_STEP(4, 4, "Executing");
    if (!tc.AllocateGraph()) return 1;
    
    ggml_backend_tensor_set(in_tensor, input.data, 0, ggml_nbytes(in_tensor));
    tc.Compute();
    
    // 5. Compare
    auto output = tc.ReadTensor(out);
    
    // For multi-stem models (like Deux with 2 stems), the output will contain all stems.
    // mask_est0.npy likely only contains the first stem (or the target stem).
    // If output size > expected size, we should compare only the matching portion (first stem).
    
    size_t expected_size = expected.nelements();
    size_t actual_size = output.size();
    
    bool pass = false;
    if (actual_size > expected_size && actual_size % expected_size == 0) {
        // De-interleave Stem 0
        // Data layout: [Freqs, Stems, Frames, Batch] (ne0, ne1, ne2, ne3)
        // Stride per frame = Freqs * Stems
        // We want Stem 0 for each frame.
        
        std::vector<float> stem0_output;
        stem0_output.reserve(expected_size);
        
        int num_stems = (int)(actual_size / expected_size);
        int n_frames = (int)input.shape[1]; // Known from input
        int n_freqs = (int)(expected_size / n_frames); // Inferred Freqs per frame
        
        std::cout << "Detected multi-stem output (" << num_stems << " stems). Verifying Stem 0..." << std::endl;
        
        // Verify assumption
        if ((size_t)(num_stems * n_freqs * n_frames) != actual_size) {
            std::cerr << "Warning: Shape mismatch calculation in verification logic." << std::endl;
        }

        for (int t = 0; t < n_frames; ++t) {
            size_t frame_start = t * (n_freqs * num_stems);
            size_t stem0_start = frame_start; // Stem 0 is at offset 0 in the stride
            
            // Copy n_freqs elements
            for (int f = 0; f < n_freqs; ++f) {
                if (stem0_start + f < output.size()) {
                    stem0_output.push_back(output[stem0_start + f]);
                }
            }
        }
        
        pass = CompareAndReport("MaskEstimator (Stem 0)",
                                expected.data, expected_size,
                                stem0_output.data(), stem0_output.size());
    } else {
        pass = CompareAndReport("MaskEstimator", 
                                expected.data, expected.nelements(),
                                output.data(), output.size());
    }
                                  
    if (pass) LOG_PASS(); else LOG_FAIL();
    return pass ? 0 : 1;
}
