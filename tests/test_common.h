#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <filesystem>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include "../src/model.h"
#include "../src/utils.h"

static constexpr int kBsrTestSkipCode = 77;

//======================================================
// 配置获取
//======================================================
inline std::string GetTestDataDir() {
    const char* env = std::getenv("BSR_TEST_DATA_DIR");
    return env ? env : ".";
}

inline std::string GetModelPath() {
    const char* env = std::getenv("BSR_MODEL_PATH");
    return env ? env : "bs_roformer.gguf";
}

inline float GetToleranceAtol() {
    const char* env = std::getenv("BSR_TEST_ATOL");
    return env ? std::stof(env) : 1e-3f;
}

inline float GetToleranceRtol() {
    const char* env = std::getenv("BSR_TEST_RTOL");
    return env ? std::stof(env) : 1e-2f;
}

//======================================================
// Skip helpers (for tests requiring external data/model)
//======================================================
inline bool PathExists(const std::string& path) {
    std::error_code ec;
    return std::filesystem::exists(std::filesystem::u8path(path), ec);
}

inline std::string ActivationPath(const std::string& dir, const std::string& name) {
    return dir + "/activations/" + name + ".npy";
}

#define TEST_SKIP(msg) \
    do { \
        std::cout << "\n[SKIP] " << msg << std::endl; \
        return kBsrTestSkipCode; \
    } while(0)

//======================================================
// RAII 测试上下文 (TestContext)
//======================================================
struct TestContext {
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    BSRoformer* model = nullptr;
    
    // 初始化上下文和图
    TestContext(BSRoformer* m, size_t mem_size = 512 * 1024 * 1024);
    
    // 析构自动释放资源
    ~TestContext();
    
    // 分配图内存 (VRAM/RAM)
    bool AllocateGraph();
    
    // 执行计算
    void Compute();
    
    // 安全读取张量数据 (自动处理 GPU->CPU 拷贝)
    std::vector<float> ReadTensor(ggml_tensor* t);
};

//======================================================
// RAII Golden Data 加载器
//======================================================
struct GoldenTensor {
    float* data = nullptr;
    std::vector<size_t> shape;
    std::string name;
    
    GoldenTensor() = default;
    
    // 从 dir/activations/{name}.npy 加载
    GoldenTensor(const std::string& dir, const std::string& name);
    
    ~GoldenTensor();
    
    // 禁止拷贝
    GoldenTensor(const GoldenTensor&) = delete;
    GoldenTensor& operator=(const GoldenTensor&) = delete;
    
    // 允许移动
    GoldenTensor(GoldenTensor&& o) noexcept;
    GoldenTensor& operator=(GoldenTensor&& o) noexcept;
    
    bool valid() const { return data != nullptr; }
    size_t nelements() const;
    
    // 打印形状
    void PrintShape(const std::string& prefix = "") const;
};

//======================================================
// 断言宏
//======================================================
#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "\n[ASSERT FAILED] " << msg << std::endl; \
            std::cerr << "  File: " << __FILE__ << ":" << __LINE__ << std::endl; \
            return 1; \
        } \
    } while(0)

#define TEST_ASSERT_LOAD(tensor, name) \
    TEST_ASSERT((tensor).valid(), "Failed to load " name ".npy from " + GetTestDataDir())

//======================================================
// 辅助函数
//======================================================

// 比较结果并打印报告
bool CompareAndReport(
    const std::string& name,
    const float* expected, size_t n_expected,
    const float* actual, size_t n_actual,
    float atol = -1.0f, // < 0 means use default/env
    float rtol = -1.0f
);

// 日志宏
#define LOG_STEP(step, total, msg) \
    std::cout << "\n[" << step << "/" << total << "] " << msg << std::endl

#define LOG_PASS() std::cout << "\n✓ PASSED" << std::endl
#define LOG_FAIL() std::cout << "\n✗ FAILED" << std::endl
