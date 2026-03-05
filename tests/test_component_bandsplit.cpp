#include "test_common.h"

int main(int argc, char* argv[]) {
    std::cout << "Test: BandSplit Component Verification" << std::endl;
    
    // 1. 获取资源
    std::string model_path = GetModelPath();
    std::string data_dir = GetTestDataDir();
    
    if (argc > 1) model_path = argv[1];
    if (argc > 2) data_dir = argv[2];

    if (!PathExists(model_path)) {
        TEST_SKIP("Model file not found: " + model_path + " (set BSR_MODEL_PATH)");
    }
    if (!PathExists(ActivationPath(data_dir, "band_split_in")) ||
        !PathExists(ActivationPath(data_dir, "after_band_split"))) {
        TEST_SKIP("Test data not found under: " + data_dir + " (set BSR_TEST_DATA_DIR)");
    }
    
    LOG_STEP(1, 4, "Loading model from " + model_path);
    BSRoformer model;
    model.Initialize(model_path);
    
    LOG_STEP(2, 4, "Loading golden tensors from " + data_dir);
    GoldenTensor input(data_dir, "band_split_in");
    GoldenTensor expected(data_dir, "after_band_split");
    
    TEST_ASSERT_LOAD(input, "band_split_in");
    TEST_ASSERT_LOAD(expected, "after_band_split");
    
    input.PrintShape("Input");
    expected.PrintShape("Expected");
    
    // PyTorch [batch, bands, time, dim] -> GGML [dim, time, bands, batch] ? 
    // Wait, utils.cpp says: load_npy returns raw data and shape.
    // PyTorch input: [batch, bands, time, dim]
    // GGML expected Input: [dim, bands, time, batch] ? No.
    // Let's check original test...
    // Original: total_dim_input(idx=2), n_frames(idx=1), batch(idx=0).
    // Original input: [batch, frames, dim] ??
    // band_split_in.npy shape from original output: [1, 301, 384] (Batch, Time, Dim)?
    // No, let's look at export_debug.py line 219: `x = rearrange(x, 'b t (f c) -> b t f c')` ??
    // Wait, export_debug.py:
    //   x = stft_repr[batch_arange, freq_indices] -> [b, f, t, c]
    //   x = rearrange(x, 'b f t c -> b t (f c)') -> [b, t, features]
    // So 'band_split_in' is [Batch, Time, Features]
    // GGML Tensor likely: [Features, Time, Batch] (Transposed for column-major/GGML)
    
    int batch = input.shape[0];
    int n_frames = input.shape[1];
    int total_dim = input.shape[2];
    
    // 3. Build Graph
    LOG_STEP(3, 4, "Building computation graph");
    TestContext tc(&model);
    
    // GGML Tensor shape: [dim, n_frames, batch]
    ggml_tensor* in_tensor = ggml_new_tensor_3d(tc.ctx, GGML_TYPE_F32, total_dim, n_frames, batch);
    ggml_set_input(in_tensor);
    
    ggml_tensor* out = model.BuildBandSplitGraph(tc.ctx, in_tensor, tc.gf, n_frames, batch);
    TEST_ASSERT(out, "BuildBandSplitGraph returned nullptr");
    
    // Mark output for computation
    ggml_build_forward_expand(tc.gf, out);
    
    // 4. Exec
    LOG_STEP(4, 4, "Executing");
    if (!tc.AllocateGraph()) {
        std::cerr << "Graph allocation failed" << std::endl;
        return 1;
    }
    
    // Copy input (NumPy [B, T, D] -> GGML [D, T, B])
    // The memory layout of NumPy [B,T,D] (C-contiguous) is:
    //   Batch 0 -> Time 0 -> Dim 0..D
    // GGML [D, T, B] (F-contiguous-ish, but tensor struct is different)
    // Actually GGML default tensor is [ne0, ne1, ne2, ne3]
    // ne0 is fastest moving dimension. 
    // If we say tensor is [D, T, B], ne0=D, ne1=T, ne2=B.
    // So data layout is D contiguous, then T, then B.
    // This MATCHES NumPy [B, T, D] C-contiguous!
    //   NumPy: fast index is last dim (D).
    //   GGML: fast index is first dim (ne0=D).
    // So we can memcpy directly!
    
    ggml_backend_tensor_set(in_tensor, input.data, 0, ggml_nbytes(in_tensor));
    tc.Compute();
    
    // 5. Compare
    auto output = tc.ReadTensor(out);
    
    bool pass = CompareAndReport("BandSplit", 
                                  expected.data, expected.nelements(),
                                  output.data(), output.size());
    
    if (pass) LOG_PASS(); else LOG_FAIL();
    return pass ? 0 : 1;
}
