#include "test_common.h"

//======================================================
// TestContext
//======================================================
TestContext::TestContext(MelBandRoformer* m, size_t mem_size) : model(m) {
    if (!model) {
        std::cerr << "FATAL: Model is null in TestContext" << std::endl;
        exit(1);
    }
    
    struct ggml_init_params ctx_params = {
        /*.mem_size   = */ mem_size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    ctx = ggml_init(ctx_params);
    gf = ggml_new_graph_custom(ctx, 16384, false); // Sufficiently large graph
}

TestContext::~TestContext() {
    if (allocr) ggml_gallocr_free(allocr);
    if (ctx) ggml_free(ctx);
}

bool TestContext::AllocateGraph() {
    if (!allocr) {
        allocr = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(model->GetBackend())
        );
    }
    return ggml_gallocr_alloc_graph(allocr, gf);
}

void TestContext::Compute() {
    ggml_backend_graph_compute(model->GetBackend(), gf);
}

std::vector<float> TestContext::ReadTensor(ggml_tensor* t) {
    size_t nelements = ggml_nelements(t);
    std::vector<float> buffer(nelements);
    ggml_backend_tensor_get(t, buffer.data(), 0, ggml_nbytes(t));
    return buffer;
}

//======================================================
// GoldenTensor
//======================================================
GoldenTensor::GoldenTensor(const std::string& dir, const std::string& n) : name(n) {
    std::pair<float*, std::vector<size_t>> res = utils::load_activation(dir, name);
    data = res.first;
    shape = res.second;
}

GoldenTensor::~GoldenTensor() {
    if (data) {
        utils::free_npy_data(data);
        data = nullptr;
    }
}

GoldenTensor::GoldenTensor(GoldenTensor&& o) noexcept 
    : data(o.data), shape(std::move(o.shape)), name(std::move(o.name)) {
    o.data = nullptr;
}

GoldenTensor& GoldenTensor::operator=(GoldenTensor&& o) noexcept {
    if (this != &o) {
        if (data) utils::free_npy_data(data);
        data = o.data;
        shape = std::move(o.shape);
        name = std::move(o.name);
        o.data = nullptr;
    }
    return *this;
}

size_t GoldenTensor::nelements() const {
    if (shape.empty()) return 0;
    size_t n = 1;
    for (size_t dim : shape) n *= dim;
    return n;
}

void GoldenTensor::PrintShape(const std::string& prefix) const {
    std::cout << prefix << name << " shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

//======================================================
// Helper
//======================================================
bool CompareAndReport(
    const std::string& name,
    const float* expected, size_t n_expected,
    const float* actual, size_t n_actual,
    float atol,
    float rtol
) {
    std::cout << "[Compare] " << name << std::endl;
    
    if (n_expected != n_actual) {
        std::cerr << "  SIZE MISMATCH: Expected " << n_expected << ", Actual " << n_actual << std::endl;
        return false;
    }
    
    // Resolve tolerances
    if (atol < 0) atol = GetToleranceAtol();
    if (rtol < 0) rtol = GetToleranceRtol();
    
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    float max_rel_diff = 0.0f;
    
    for (size_t i = 0; i < n_expected; ++i) {
        float diff = std::abs(expected[i] - actual[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        
        if (std::abs(expected[i]) > 1e-8f) {
            float rel = diff / std::abs(expected[i]);
            max_rel_diff = std::max(max_rel_diff, rel);
        }
    }
    
    float mean_diff = sum_diff / n_expected;
    
    std::cout << "  max_diff: " << max_diff << " (limit " << atol << ")" << std::endl;
    std::cout << "  mean_diff: " << mean_diff << std::endl;
    std::cout << "  max_rel_diff: " << max_rel_diff << " (limit " << rtol << ")" << std::endl;
    
    bool match = (max_diff <= atol) || (max_rel_diff <= rtol);
    
    if (match) {
        std::cout << "  ✓ OK" << std::endl;
    } else {
        std::cout << "  ✗ MISMATCH" << std::endl;
    }
    
    return match;
}
