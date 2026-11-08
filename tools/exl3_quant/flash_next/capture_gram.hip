// Diagnostic only. Original FP16 values, FP32 accumulation; no precision casts.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdint>
#include <new>
#include <initializer_list>

using H16 = _Float16 __attribute__((ext_vector_type(16)));
using F8 = float __attribute__((ext_vector_type(8)));

__global__ void gram_wmma(const _Float16* x, float* y, int rows, int dim) {
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x / 32;
    const int tiles = (dim + 15) / 16;
    const int tile = blockIdx.x * 4 + wave;
    if (tile >= tiles * tiles) return; // entire wave exits together
    const int i0 = (tile / tiles) * 16, j0 = (tile % tiles) * 16;
    const int a_col = i0 + lane % 16, b_col = j0 + lane % 16;
    const int64_t base = int64_t(blockIdx.y) * rows * dim;
    F8 acc = {};
    for (int k = 0; k < rows; k += 16) {
        H16 a = {}, b = {};
        #pragma unroll
        for (int q = 0; q < 16; ++q) {
            if (k + q < rows && a_col < dim) a[q] = x[base + int64_t(k + q) * dim + a_col];
            if (k + q < rows && b_col < dim) b[q] = x[base + int64_t(k + q) * dim + b_col];
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    }
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        const int i = i0 + 2 * g + lane / 16, j = j0 + lane % 16;
        if (i < dim && j < dim) y[(int64_t(blockIdx.y) * dim + i) * dim + j] = acc[g];
    }
}
extern "C" int gram_launch(const void* x, void* y, int batch, int rows, int dim, void* stream) {
    if (!x || !y || batch < 1 || rows < 1 || dim < 1) return -1;
    int tiles = (dim + 15) / 16;
    hipLaunchKernelGGL(gram_wmma, dim3((tiles * tiles + 3) / 4, batch), dim3(128), 0,
                      reinterpret_cast<hipStream_t>(stream), static_cast<const _Float16*>(x),
                      static_cast<float*>(y), rows, dim);
    return int(hipGetLastError());
}

struct Plan {
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    hipblasLtMatmulHeuristicResult_t algorithms[4]{};
    int found = 0;
    size_t workspace_bytes = 32 * 1024 * 1024;
    void* workspace = nullptr;
    ~Plan() {
        if (workspace) hipFree(workspace);
        if (pref) hipblasLtMatmulPreferenceDestroy(pref);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
        if (op) hipblasLtMatmulDescDestroy(op);
        if (handle) hipblasLtDestroy(handle);
    }
};
#define CHECK(call) do { int code = int(call); if (code) { *status = code; delete p; return nullptr; } } while (0)
extern "C" void* gram_lt_create(int batch, int rows, int dim, int half, int* status, int* found) {
    *status = -1; *found = 0;
    if (batch < 1 || rows < 1 || dim < 1 || (half != 0 && half != 1)) return nullptr;
    auto* p = new(std::nothrow) Plan;
    if (!p) return nullptr;
    CHECK(hipblasLtCreate(&p->handle));
    CHECK(hipblasLtMatmulDescCreate(&p->op, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t trans = HIPBLAS_OP_T;
    CHECK(hipblasLtMatmulDescSetAttribute(p->op, HIPBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));
    hipDataType dtype = half ? HIP_R_16F : HIP_R_32F;
    CHECK(hipblasLtMatrixLayoutCreate(&p->a, dtype, dim, rows, dim));
    CHECK(hipblasLtMatrixLayoutCreate(&p->b, dtype, dim, rows, dim));
    CHECK(hipblasLtMatrixLayoutCreate(&p->c, HIP_R_32F, dim, dim, dim));
    int32_t count = batch;
    int64_t sx = int64_t(rows) * dim, sy = int64_t(dim) * dim;
    for (auto layout : {p->a, p->b, p->c}) {
        CHECK(hipblasLtMatrixLayoutSetAttribute(layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)));
        int64_t stride = layout == p->c ? sy : sx;
        CHECK(hipblasLtMatrixLayoutSetAttribute(layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
    }
    CHECK(hipblasLtMatmulPreferenceCreate(&p->pref));
    CHECK(hipblasLtMatmulPreferenceSetAttribute(p->pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &p->workspace_bytes, sizeof(p->workspace_bytes)));
    CHECK(hipblasLtMatmulAlgoGetHeuristic(p->handle, p->op, p->a, p->b, p->c, p->c, p->pref, 4, p->algorithms, &p->found));
    if (!p->found) { *status = -2; delete p; return nullptr; }
    CHECK(hipMalloc(&p->workspace, p->workspace_bytes));
    *status = 0; *found = p->found; return p;
}
extern "C" int gram_lt_run(void* plan, int algorithm, const void* x, void* y, void* stream) {
    auto* p = static_cast<Plan*>(plan);
    if (!p || algorithm < 0 || algorithm >= p->found || !x || !y) return -1;
    const float alpha = 1.0f, beta = 0.0f;
    return int(hipblasLtMatmul(p->handle, p->op, &alpha, x, p->a, x, p->b, &beta,
                             y, p->c, y, p->c, &p->algorithms[algorithm].algo,
                             p->workspace, p->workspace_bytes, reinterpret_cast<hipStream_t>(stream)));
}
extern "C" void gram_lt_destroy(void* plan) { delete static_cast<Plan*>(plan); }
