// Operator candidates only: original half operands, float accumulation/output.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdint>
#include <new>

struct ForwardPlan {
    hipblasLtHandle_t h=nullptr;
    hipblasLtMatmulDesc_t op=nullptr;
    hipblasLtMatrixLayout_t w=nullptr,x=nullptr,y=nullptr;
    hipblasLtMatmulPreference_t pref=nullptr;
    hipblasLtMatmulHeuristicResult_t algorithms[8]{};
    int found=0;
    size_t bytes=32*1024*1024;
    void* workspace=nullptr;
    ~ForwardPlan() {
        if(workspace) hipFree(workspace);
        if(pref) hipblasLtMatmulPreferenceDestroy(pref);
        if(w) hipblasLtMatrixLayoutDestroy(w);
        if(x) hipblasLtMatrixLayoutDestroy(x);
        if(y) hipblasLtMatrixLayoutDestroy(y);
        if(op) hipblasLtMatmulDescDestroy(op);
        if(h) hipblasLtDestroy(h);
    }
};
#define CHECK(call) do {int rc=int(call);if(rc){*status=rc;delete p;return nullptr;}}while(0)
extern "C" void* forward_create(int batch,int m,int k,int n,int output_half,int* status,int* found) {
    *status=-1;*found=0;
    if(batch<1||m<1||k<1||n<1||(output_half!=0&&output_half!=1))return nullptr;
    auto* p=new(std::nothrow) ForwardPlan;if(!p)return nullptr;
    CHECK(hipblasLtCreate(&p->h));
    CHECK(hipblasLtMatmulDescCreate(&p->op,HIPBLAS_COMPUTE_32F,HIP_R_32F));
    // Row-major X[M,K] @ W[B,K,N] => column-major W'[N,K] @ X'[K,M].
    CHECK(hipblasLtMatrixLayoutCreate(&p->w,HIP_R_16F,n,k,n));
    CHECK(hipblasLtMatrixLayoutCreate(&p->x,HIP_R_16F,k,m,k));
    CHECK(hipblasLtMatrixLayoutCreate(&p->y,output_half?HIP_R_16F:HIP_R_32F,n,m,n));
    hipblasLtMatrixLayout_t layouts[]={p->w,p->x,p->y};
    int64_t strides[]={int64_t(k)*n,0,int64_t(m)*n};
    int32_t count=batch;
    for(int i=0;i<3;++i){
        CHECK(hipblasLtMatrixLayoutSetAttribute(layouts[i],HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&count,sizeof(count)));
        CHECK(hipblasLtMatrixLayoutSetAttribute(layouts[i],HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&strides[i],sizeof(int64_t)));
    }
    CHECK(hipblasLtMatmulPreferenceCreate(&p->pref));
    CHECK(hipblasLtMatmulPreferenceSetAttribute(p->pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&p->bytes,sizeof(p->bytes)));
    CHECK(hipblasLtMatmulAlgoGetHeuristic(p->h,p->op,p->w,p->x,p->y,p->y,p->pref,8,p->algorithms,&p->found));
    if(!p->found){*status=-2;delete p;return nullptr;}
    CHECK(hipMalloc(&p->workspace,p->bytes));
    *found=p->found;*status=0;return p;
}
extern "C" int forward_run(void* plan,int algorithm,const void* x,const void* w,void* y,void* stream){
    auto* p=static_cast<ForwardPlan*>(plan);
    if(!p||algorithm<0||algorithm>=p->found||!x||!w||!y)return -1;
    float alpha=1,beta=0;
    return int(hipblasLtMatmul(p->h,p->op,&alpha,w,p->w,x,p->x,&beta,y,p->y,y,p->y,
        &p->algorithms[algorithm].algo,p->workspace,p->bytes,reinterpret_cast<hipStream_t>(stream)));
}
extern "C" void forward_destroy(void* plan){delete static_cast<ForwardPlan*>(plan);}

using H16=_Float16 __attribute__((ext_vector_type(16)));
using F8=float __attribute__((ext_vector_type(8)));
__global__ void forward_wmma(const _Float16* x,const _Float16* w,void* y,int m,int k,int n,int output_half){
    int lane=threadIdx.x&31,wave=threadIdx.x/32;
    int nt=(n+15)/16,mt=(m+15)/16,tile=blockIdx.x*4+wave;
    if(tile>=mt*nt)return;
    int i0=tile/nt*16,j0=tile%nt*16,ir=i0+lane%16,jc=j0+lane%16;
    int64_t wb=int64_t(blockIdx.y)*k*n,yb=int64_t(blockIdx.y)*m*n;
    F8 acc={};
    for(int kk=0;kk<k;kk+=16){
        H16 a={},b={};
        #pragma unroll
        for(int q=0;q<16;++q){
            if(ir<m&&kk+q<k)a[q]=x[int64_t(ir)*k+kk+q];
            if(jc<n&&kk+q<k)b[q]=w[wb+int64_t(kk+q)*n+jc];
        }
        acc=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b,acc);
    }
    #pragma unroll
    for(int g=0;g<8;++g){
        int i=i0+2*g+lane/16,j=j0+lane%16;
        if(i<m&&j<n){
            int64_t offset=yb+int64_t(i)*n+j;
            if(output_half)static_cast<_Float16*>(y)[offset]=_Float16(acc[g]);
            else static_cast<float*>(y)[offset]=acc[g];
        }
    }
}
extern "C" int forward_wmma_run(const void* x,const void* w,void* y,int batch,int m,int k,int n,int output_half,void* stream){
    if(!x||!w||!y||batch<1||m<1||k<1||n<1||(output_half!=0&&output_half!=1))return -1;
    int tiles=((m+15)/16)*((n+15)/16);
    hipLaunchKernelGGL(forward_wmma,dim3((tiles+3)/4,batch),dim3(128),0,
        reinterpret_cast<hipStream_t>(stream),static_cast<const _Float16*>(x),
        static_cast<const _Float16*>(w),y,m,k,n,output_half);
    return int(hipGetLastError());
}
