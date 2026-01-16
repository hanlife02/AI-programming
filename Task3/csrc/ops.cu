#include <torch/extension.h>

#include <cudnn.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <tuple>

namespace {

#define CUDNN_CHECK(expr)                                                                             \
    do {                                                                                              \
        cudnnStatus_t status = (expr);                                                                \
        TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cuDNN error: ", cudnnGetErrorString(status));    \
    } while (0)

inline void check_cuda(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == at::kFloat || t.scalar_type() == at::kInt || t.scalar_type() == at::kLong,
                name, " must be float32/int32/int64");
}

inline void check_float_cuda(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
}

__device__ __forceinline__ float relu(float x) { return x > 0.0f ? x : 0.0f; }

struct CudnnHandle {
    cudnnHandle_t handle;
    CudnnHandle() { CUDNN_CHECK(cudnnCreate(&handle)); }
    ~CudnnHandle() { cudnnDestroy(handle); }
    CudnnHandle(const CudnnHandle&) = delete;
    CudnnHandle& operator=(const CudnnHandle&) = delete;
};

struct CudnnTensorDesc {
    cudnnTensorDescriptor_t desc;
    CudnnTensorDesc() { CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc)); }
    ~CudnnTensorDesc() { cudnnDestroyTensorDescriptor(desc); }
    CudnnTensorDesc(const CudnnTensorDesc&) = delete;
    CudnnTensorDesc& operator=(const CudnnTensorDesc&) = delete;
};

struct CudnnFilterDesc {
    cudnnFilterDescriptor_t desc;
    CudnnFilterDesc() { CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc)); }
    ~CudnnFilterDesc() { cudnnDestroyFilterDescriptor(desc); }
    CudnnFilterDesc(const CudnnFilterDesc&) = delete;
    CudnnFilterDesc& operator=(const CudnnFilterDesc&) = delete;
};

struct CudnnConvDesc {
    cudnnConvolutionDescriptor_t desc;
    CudnnConvDesc() { CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc)); }
    ~CudnnConvDesc() { cudnnDestroyConvolutionDescriptor(desc); }
    CudnnConvDesc(const CudnnConvDesc&) = delete;
    CudnnConvDesc& operator=(const CudnnConvDesc&) = delete;
};

inline cudnnHandle_t get_cudnn_handle(cudaStream_t stream) {
    static thread_local CudnnHandle handle;
    CUDNN_CHECK(cudnnSetStream(handle.handle, stream));
    return handle.handle;
}

inline void set_tensor4d_desc(cudnnTensorDescriptor_t desc, int N, int C, int H, int W) {
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
}

inline void set_filter4d_desc(cudnnFilterDescriptor_t desc, int K, int C, int H, int W) {
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, H, W));
}

inline bool cudnn_enabled() {
    const char* env = std::getenv("TASK3_USE_CUDNN");
    if (!env) return true;
    return std::string(env) != "0";
}

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x & (warpSize - 1);
    int wid = threadIdx.x >> 5;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void conv2d_fwd_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                  const float* __restrict__ b, float* __restrict__ y, int N, int Cin, int Hin,
                                  int Win, int Cout, int Kh, int Kw, int Hout, int Wout, int stride,
                                  int padding, bool has_bias) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;
    int n = nc / Cout;
    int oc = nc - n * Cout;
    if (n >= N || oc >= Cout || oh >= Hout || ow >= Wout) return;

    float acc = has_bias ? b[oc] : 0.0f;
    int ih0 = oh * stride - padding;
    int iw0 = ow * stride - padding;

    const int w_oc_offset = ((oc * Cin) * Kh) * Kw;
    const int x_n_offset = ((n * Cin) * Hin) * Win;
    for (int ic = 0; ic < Cin; ++ic) {
        const float* w_ptr = w + w_oc_offset + (ic * Kh * Kw);
        const float* x_ptr = x + x_n_offset + (ic * Hin * Win);
        #pragma unroll
        for (int kh = 0; kh < 7; ++kh) {  // hard cap for small kernels; guarded by kh < Kh below
            if (kh >= Kh) break;
            int ih = ih0 + kh;
            if ((unsigned)ih >= (unsigned)Hin) continue;
            #pragma unroll
            for (int kw = 0; kw < 7; ++kw) {
                if (kw >= Kw) break;
                int iw = iw0 + kw;
                if ((unsigned)iw >= (unsigned)Win) continue;
                acc += x_ptr[ih * Win + iw] * w_ptr[kh * Kw + kw];
            }
        }
    }
    y[(((n * Cout + oc) * Hout) + oh) * Wout + ow] = acc;
}

template <int OC_TILE>
__global__ void conv2d_fwd_kernel_oc_tile(const float* __restrict__ x, const float* __restrict__ w,
                                         const float* __restrict__ b, float* __restrict__ y, int N, int Cin, int Hin,
                                         int Win, int Cout, int Kh, int Kw, int Hout, int Wout, int stride,
                                         int padding, bool has_bias, int oc_tiles) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int nt = blockIdx.z;
    int n = nt / oc_tiles;
    int oc0 = (nt - n * oc_tiles) * OC_TILE;
    if (n >= N || oh >= Hout || ow >= Wout) return;

    int ih0 = oh * stride - padding;
    int iw0 = ow * stride - padding;
    const int x_n_offset = ((n * Cin) * Hin) * Win;

    for (int oc = oc0; oc < oc0 + OC_TILE && oc < Cout; ++oc) {
        float acc = (has_bias && b) ? b[oc] : 0.0f;
        const int w_oc_offset = ((oc * Cin) * Kh) * Kw;
        for (int ic = 0; ic < Cin; ++ic) {
            const float* w_ptr = w + w_oc_offset + (ic * Kh * Kw);
            const float* x_ptr = x + x_n_offset + (ic * Hin * Win);
            #pragma unroll
            for (int kh = 0; kh < 7; ++kh) {  // hard cap for small kernels; guarded by kh < Kh below
                if (kh >= Kh) break;
                int ih = ih0 + kh;
                if ((unsigned)ih >= (unsigned)Hin) continue;
                #pragma unroll
                for (int kw = 0; kw < 7; ++kw) {
                    if (kw >= Kw) break;
                    int iw = iw0 + kw;
                    if ((unsigned)iw >= (unsigned)Win) continue;
                    acc += x_ptr[ih * Win + iw] * w_ptr[kh * Kw + kw];
                }
            }
        }
        y[(((n * Cout + oc) * Hout) + oh) * Wout + ow] = acc;
    }
}

template <int TILE_H, int TILE_W, int TILE_C>
__global__ void conv2d_fwd_3x3s1p1_tiled_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                const float* __restrict__ b, float* __restrict__ y, int N, int Cin,
                                                int Hin, int Win, int Cout, int Hout, int Wout, bool has_bias) {
    static_assert(TILE_H > 0 && TILE_W > 0 && TILE_C > 0, "invalid tile sizes");
    constexpr int Kh = 3;
    constexpr int Kw = 3;
    constexpr int PAD = 1;

    int ow = blockIdx.x * TILE_W + threadIdx.x;
    int oh = blockIdx.y * TILE_H + threadIdx.y;
    int nc = blockIdx.z;
    int n = nc / Cout;
    int oc = nc - n * Cout;

    float acc = 0.0f;
    if (n < N && oc < Cout && oh < Hout && ow < Wout && has_bias) acc = b[oc];

    __shared__ float s_x[TILE_C][TILE_H + 2][TILE_W + 2];
    __shared__ float s_w[TILE_C][Kh][Kw];

    int tid = threadIdx.y * TILE_W + threadIdx.x;

    for (int ic0 = 0; ic0 < Cin; ic0 += TILE_C) {
        // Load weights [TILE_C, 3, 3]
        for (int idx = tid; idx < TILE_C * Kh * Kw; idx += TILE_H * TILE_W) {
            int t = idx;
            int c = t / (Kh * Kw);
            t -= c * (Kh * Kw);
            int kh = t / Kw;
            int kw = t - kh * Kw;
            int ic = ic0 + c;
            float v = 0.0f;
            if (oc < Cout && ic < Cin) {
                v = w[(((oc * Cin + ic) * Kh) + kh) * Kw + kw];
            }
            s_w[c][kh][kw] = v;
        }

        // Load input tile [TILE_C, TILE_H+2, TILE_W+2]
        constexpr int TILE_IN_H = TILE_H + 2;
        constexpr int TILE_IN_W = TILE_W + 2;
        constexpr int TILE_IN_HW = TILE_IN_H * TILE_IN_W;
        for (int idx = tid; idx < TILE_C * TILE_IN_HW; idx += TILE_H * TILE_W) {
            int c = idx / TILE_IN_HW;
            int rem = idx - c * TILE_IN_HW;
            int th = rem / TILE_IN_W;
            int tw = rem - th * TILE_IN_W;
            int ic = ic0 + c;
            int ih = blockIdx.y * TILE_H + th - PAD;
            int iw = blockIdx.x * TILE_W + tw - PAD;
            float v = 0.0f;
            if (n < N && ic < Cin && (unsigned)ih < (unsigned)Hin && (unsigned)iw < (unsigned)Win) {
                v = x[(((n * Cin + ic) * Hin) + ih) * Win + iw];
            }
            s_x[c][th][tw] = v;
        }
        __syncthreads();

        if (n < N && oc < Cout && oh < Hout && ow < Wout) {
            #pragma unroll
            for (int c = 0; c < TILE_C; ++c) {
                int ic = ic0 + c;
                if (ic >= Cin) break;
                float x00 = s_x[c][threadIdx.y + 0][threadIdx.x + 0];
                float x01 = s_x[c][threadIdx.y + 0][threadIdx.x + 1];
                float x02 = s_x[c][threadIdx.y + 0][threadIdx.x + 2];
                float x10 = s_x[c][threadIdx.y + 1][threadIdx.x + 0];
                float x11 = s_x[c][threadIdx.y + 1][threadIdx.x + 1];
                float x12 = s_x[c][threadIdx.y + 1][threadIdx.x + 2];
                float x20 = s_x[c][threadIdx.y + 2][threadIdx.x + 0];
                float x21 = s_x[c][threadIdx.y + 2][threadIdx.x + 1];
                float x22 = s_x[c][threadIdx.y + 2][threadIdx.x + 2];

                float w00 = s_w[c][0][0];
                float w01 = s_w[c][0][1];
                float w02 = s_w[c][0][2];
                float w10 = s_w[c][1][0];
                float w11 = s_w[c][1][1];
                float w12 = s_w[c][1][2];
                float w20 = s_w[c][2][0];
                float w21 = s_w[c][2][1];
                float w22 = s_w[c][2][2];

                acc += x00 * w00 + x01 * w01 + x02 * w02;
                acc += x10 * w10 + x11 * w11 + x12 * w12;
                acc += x20 * w20 + x21 * w21 + x22 * w22;
            }
        }
        __syncthreads();
    }

    if (n < N && oc < Cout && oh < Hout && ow < Wout) {
        y[(((n * Cout + oc) * Hout) + oh) * Wout + ow] = acc;
    }
}

__global__ void conv2d_bwd_input_kernel(const float* __restrict__ grad_out, const float* __restrict__ w,
                                        float* __restrict__ grad_x, int N, int Cin, int Hin, int Win, int Cout,
                                        int Kh, int Kw, int Hout, int Wout, int stride, int padding) {
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;
    int n = nc / Cin;
    int ic = nc - n * Cin;
    if (n >= N || ic >= Cin || ih >= Hin || iw >= Win) return;

    float acc = 0.0f;
    for (int oc = 0; oc < Cout; ++oc) {
        const float* w_ptr = w + (((oc * Cin + ic) * Kh) * Kw);
        const float* go_ptr = grad_out + (((n * Cout + oc) * Hout) * Wout);
        for (int kh = 0; kh < Kh; ++kh) {
            int oh = ih + padding - kh;
            if (oh % stride != 0) continue;
            oh /= stride;
            if ((unsigned)oh >= (unsigned)Hout) continue;
            for (int kw = 0; kw < Kw; ++kw) {
                int ow = iw + padding - kw;
                if (ow % stride != 0) continue;
                ow /= stride;
                if ((unsigned)ow >= (unsigned)Wout) continue;
                acc += go_ptr[oh * Wout + ow] * w_ptr[kh * Kw + kw];
            }
        }
    }
    grad_x[(((n * Cin + ic) * Hin) + ih) * Win + iw] = acc;
}

template <int IC_TILE>
__global__ void conv2d_bwd_input_kernel_ic_tile(const float* __restrict__ grad_out, const float* __restrict__ w,
                                               float* __restrict__ grad_x, int N, int Cin, int Hin, int Win, int Cout,
                                               int Kh, int Kw, int Hout, int Wout, int stride, int padding,
                                               int ic_tiles) {
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int nt = blockIdx.z;
    int n = nt / ic_tiles;
    int ic0 = (nt - n * ic_tiles) * IC_TILE;
    if (n >= N || ih >= Hin || iw >= Win) return;

    for (int ic = ic0; ic < ic0 + IC_TILE && ic < Cin; ++ic) {
        float acc = 0.0f;
        for (int oc = 0; oc < Cout; ++oc) {
            const float* w_ptr = w + (((oc * Cin + ic) * Kh) * Kw);
            const float* go_ptr = grad_out + (((n * Cout + oc) * Hout) * Wout);
            for (int kh = 0; kh < Kh; ++kh) {
                int oh = ih + padding - kh;
                if (oh % stride != 0) continue;
                oh /= stride;
                if ((unsigned)oh >= (unsigned)Hout) continue;
                for (int kw = 0; kw < Kw; ++kw) {
                    int ow = iw + padding - kw;
                    if (ow % stride != 0) continue;
                    ow /= stride;
                    if ((unsigned)ow >= (unsigned)Wout) continue;
                    acc += go_ptr[oh * Wout + ow] * w_ptr[kh * Kw + kw];
                }
            }
        }
        grad_x[(((n * Cin + ic) * Hin) + ih) * Win + iw] = acc;
    }
}

template <int THREADS>
__global__ void conv2d_bwd_weight_kernel(const float* __restrict__ grad_out, const float* __restrict__ x,
                                         float* __restrict__ grad_w, int N, int Cin, int Hin, int Win, int Cout,
                                         int Kh, int Kw, int Hout, int Wout, int stride, int padding) {
    int oc = blockIdx.x;
    int ic = blockIdx.y;
    int k = blockIdx.z;
    int kh = k / Kw;
    int kw = k - kh * Kw;

    float sum = 0.0f;
    int total = N * Hout * Wout;
    for (int idx = threadIdx.x; idx < total; idx += THREADS) {
        int tmp = idx;
        int ow = tmp % Wout;
        tmp /= Wout;
        int oh = tmp % Hout;
        int n = tmp / Hout;

        int ih = oh * stride - padding + kh;
        int iw = ow * stride - padding + kw;
        if ((unsigned)ih >= (unsigned)Hin || (unsigned)iw >= (unsigned)Win) continue;
        float xv = x[(((n * Cin + ic) * Hin) + ih) * Win + iw];
        float gov = grad_out[(((n * Cout + oc) * Hout) + oh) * Wout + ow];
        sum += xv * gov;
    }

    __shared__ float buf[THREADS];
    buf[threadIdx.x] = sum;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        grad_w[(((oc * Cin + ic) * Kh) + kh) * Kw + kw] = buf[0];
    }
}

template <int THREADS>
__global__ void conv2d_bwd_bias_kernel(const float* __restrict__ grad_out, float* __restrict__ grad_b, int N,
                                       int Cout, int Hout, int Wout) {
    int oc = blockIdx.x;
    float sum = 0.0f;
    int total = N * Hout * Wout;
    const float* go = grad_out + oc * Hout * Wout;
    for (int idx = threadIdx.x; idx < total; idx += THREADS) {
        int tmp = idx;
        int ow = tmp % Wout;
        tmp /= Wout;
        int oh = tmp % Hout;
        int n = tmp / Hout;
        sum += go[((n * Cout) * Hout + oh) * Wout + ow];
    }
    __shared__ float buf[THREADS];
    buf[threadIdx.x] = sum;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) grad_b[oc] = buf[0];
}

__global__ void relu_fwd_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = relu(x[i]);
}

__global__ void relu_bwd_kernel(const float* __restrict__ grad_out, const float* __restrict__ x,
                                float* __restrict__ grad_x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    grad_x[i] = x[i] > 0.0f ? grad_out[i] : 0.0f;
}

__global__ void relu_fwd_vec4_kernel(const float* __restrict__ x, float* __restrict__ y, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int off = idx * 4;
    if (off >= n4) return;
    float4 v = reinterpret_cast<const float4*>(x)[idx];
    v.x = relu(v.x);
    v.y = relu(v.y);
    v.z = relu(v.z);
    v.w = relu(v.w);
    reinterpret_cast<float4*>(y)[idx] = v;
}

__global__ void relu_bwd_vec4_kernel(const float* __restrict__ grad_out, const float* __restrict__ x,
                                     float* __restrict__ grad_x, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int off = idx * 4;
    if (off >= n4) return;
    float4 go = reinterpret_cast<const float4*>(grad_out)[idx];
    float4 xv = reinterpret_cast<const float4*>(x)[idx];
    float4 gx;
    gx.x = xv.x > 0.0f ? go.x : 0.0f;
    gx.y = xv.y > 0.0f ? go.y : 0.0f;
    gx.z = xv.z > 0.0f ? go.z : 0.0f;
    gx.w = xv.w > 0.0f ? go.w : 0.0f;
    reinterpret_cast<float4*>(grad_x)[idx] = gx;
}

__global__ void maxpool2d_fwd_kernel(const float* __restrict__ x, float* __restrict__ y, int32_t* __restrict__ idx,
                                     int N, int C, int Hin, int Win, int Hout, int Wout, int kernel, int stride) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc - n * C;
    if (n >= N || c >= C || oh >= Hout || ow >= Wout) return;

    int ih0 = oh * stride;
    int iw0 = ow * stride;

    float best = -INFINITY;
    int best_i = 0;
    const float* x_ptr = x + (((n * C + c) * Hin) * Win);
    for (int kh = 0; kh < kernel; ++kh) {
        int ih = ih0 + kh;
        if (ih >= Hin) continue;
        for (int kw = 0; kw < kernel; ++kw) {
            int iw = iw0 + kw;
            if (iw >= Win) continue;
            float v = x_ptr[ih * Win + iw];
            if (v > best) {
                best = v;
                best_i = ih * Win + iw;
            }
        }
    }
    y[(((n * C + c) * Hout) + oh) * Wout + ow] = best;
    idx[(((n * C + c) * Hout) + oh) * Wout + ow] = best_i;
}

template <int C_TILE>
__global__ void maxpool2d_fwd_kernel_c_tile(const float* __restrict__ x, float* __restrict__ y,
                                           int32_t* __restrict__ idx, int N, int C, int Hin, int Win, int Hout,
                                           int Wout, int kernel, int stride, int c_tiles) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int nt = blockIdx.z;
    int n = nt / c_tiles;
    int c0 = (nt - n * c_tiles) * C_TILE;
    if (n >= N || oh >= Hout || ow >= Wout) return;

    int ih0 = oh * stride;
    int iw0 = ow * stride;
    for (int c = c0; c < c0 + C_TILE && c < C; ++c) {
        float best = -INFINITY;
        int best_i = 0;
        const float* x_ptr = x + (((n * C + c) * Hin) * Win);
        for (int kh = 0; kh < kernel; ++kh) {
            int ih = ih0 + kh;
            if (ih >= Hin) continue;
            for (int kw = 0; kw < kernel; ++kw) {
                int iw = iw0 + kw;
                if (iw >= Win) continue;
                float v = x_ptr[ih * Win + iw];
                if (v > best) {
                    best = v;
                    best_i = ih * Win + iw;
                }
            }
        }
        y[(((n * C + c) * Hout) + oh) * Wout + ow] = best;
        idx[(((n * C + c) * Hout) + oh) * Wout + ow] = best_i;
    }
}

__global__ void maxpool2d_bwd_kernel(const float* __restrict__ grad_out, const int32_t* __restrict__ idx,
                                     float* __restrict__ grad_x, int N, int C, int Hin, int Win, int Hout,
                                     int Wout) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc - n * C;
    if (n >= N || c >= C || oh >= Hout || ow >= Wout) return;

    int out_offset = (((n * C + c) * Hout) + oh) * Wout + ow;
    int in_i = idx[out_offset];
    float go = grad_out[out_offset];
    grad_x[(((n * C + c) * Hin) * Win) + in_i] = go;
}

template <int C_TILE>
__global__ void maxpool2d_bwd_kernel_c_tile(const float* __restrict__ grad_out, const int32_t* __restrict__ idx,
                                           float* __restrict__ grad_x, int N, int C, int Hin, int Win, int Hout,
                                           int Wout, int c_tiles) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int nt = blockIdx.z;
    int n = nt / c_tiles;
    int c0 = (nt - n * c_tiles) * C_TILE;
    if (n >= N || oh >= Hout || ow >= Wout) return;

    for (int c = c0; c < c0 + C_TILE && c < C; ++c) {
        int out_offset = (((n * C + c) * Hout) + oh) * Wout + ow;
        int in_i = idx[out_offset];
        float go = grad_out[out_offset];
        grad_x[(((n * C + c) * Hin) * Win) + in_i] = go;
    }
}

__global__ void global_avg_pool_fwd_kernel(const float* __restrict__ x, float* __restrict__ y, int N, int C,
                                           int H, int W) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if (n >= N || c >= C) return;
    const float* x_ptr = x + (((n * C + c) * H) * W);
    float sum = 0.0f;
    int HW = H * W;
    for (int i = 0; i < HW; ++i) sum += x_ptr[i];
    y[n * C + c] = sum / (float)HW;
}

__global__ void global_avg_pool_bwd_kernel(const float* __restrict__ grad_out, float* __restrict__ grad_x, int N,
                                           int C, int H, int W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (i >= total) return;
    int W_idx = i % W;
    (void)W_idx;
    int tmp = i / W;
    int H_idx = tmp % H;
    (void)H_idx;
    tmp /= H;
    int c = tmp % C;
    int n = tmp / C;
    float go = grad_out[n * C + c];
    grad_x[i] = go / (float)(H * W);
}

__global__ void linear_fwd_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                  const float* __restrict__ b, float* __restrict__ y, int N, int In, int Out,
                                  bool has_bias) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if (n >= N || o >= Out) return;
    const float* x_ptr = x + n * In;
    const float* w_ptr = w + o * In;
    float acc = has_bias ? b[o] : 0.0f;
    for (int i = 0; i < In; ++i) acc += x_ptr[i] * w_ptr[i];
    y[n * Out + o] = acc;
}

template <int TILE>
__global__ void linear_fwd_tiled_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                        const float* __restrict__ b, float* __restrict__ y, int N, int In, int Out,
                                        bool has_bias) {
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ float s_x[TILE][TILE];
    __shared__ float s_w[TILE][TILE];

    float acc = 0.0f;
    if (has_bias && row < N && col < Out) acc = b[col];

    int tiles = (In + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        int kx = t * TILE + threadIdx.x;
        int kw = t * TILE + threadIdx.y;

        s_x[threadIdx.y][threadIdx.x] = (row < N && kx < In) ? x[row * In + kx] : 0.0f;
        // w is [Out, In]; load it as B[K, Out] where B[k, col] = w[col, k]
        s_w[threadIdx.y][threadIdx.x] = (col < Out && kw < In) ? w[col * In + kw] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += s_x[threadIdx.y][i] * s_w[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < Out) y[row * Out + col] = acc;
}

__global__ void linear_bwd_input_kernel(const float* __restrict__ grad_out, const float* __restrict__ w,
                                        float* __restrict__ grad_x, int N, int In, int Out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if (n >= N || i >= In) return;
    float acc = 0.0f;
    for (int o = 0; o < Out; ++o) acc += grad_out[n * Out + o] * w[o * In + i];
    grad_x[n * In + i] = acc;
}

template <int THREADS>
__global__ void linear_bwd_weight_kernel(const float* __restrict__ grad_out, const float* __restrict__ x,
                                         float* __restrict__ grad_w, int N, int In, int Out) {
    int o = blockIdx.x;
    int i = blockIdx.y;
    float sum = 0.0f;
    for (int n = threadIdx.x; n < N; n += THREADS) sum += grad_out[n * Out + o] * x[n * In + i];
    __shared__ float buf[THREADS];
    buf[threadIdx.x] = sum;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) grad_w[o * In + i] = buf[0];
}

template <int THREADS>
__global__ void linear_bwd_bias_kernel(const float* __restrict__ grad_out, float* __restrict__ grad_b, int N,
                                       int Out) {
    int o = blockIdx.x;
    float sum = 0.0f;
    for (int n = threadIdx.x; n < N; n += THREADS) sum += grad_out[n * Out + o];
    __shared__ float buf[THREADS];
    buf[threadIdx.x] = sum;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) grad_b[o] = buf[0];
}

__global__ void bn_stats_kernel(const float* __restrict__ x, float* __restrict__ mean, float* __restrict__ var,
                                int N, int C, int H, int W) {
    int c = blockIdx.x;
    if (c >= C) return;
    int HW = H * W;
    int M = N * HW;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        int n = i / HW;
        int hw = i - n * HW;
        float v = x[((n * C + c) * HW) + hw];
        sum += v;
        sumsq += v * v;
    }

    __shared__ float shared_sum[32];
    __shared__ float shared_sumsq[32];
    float s0 = blockReduceSum(sum, shared_sum);
    float s1 = blockReduceSum(sumsq, shared_sumsq);
    if (threadIdx.x == 0) {
        float m = s0 / (float)M;
        float vv = s1 / (float)M - m * m;
        mean[c] = m;
        var[c] = vv;
    }
}

__global__ void bn_fwd_kernel(const float* __restrict__ x, const float* __restrict__ weight,
                              const float* __restrict__ bias, const float* __restrict__ mean,
                              const float* __restrict__ invstd, float* __restrict__ y, int N, int C, int H, int W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (i >= total) return;
    int HW = H * W;
    int tmp = i / HW;
    int c = tmp % C;
    float xv = x[i];
    float xhat = (xv - mean[c]) * invstd[c];
    y[i] = xhat * weight[c] + bias[c];
}

__global__ void bn_dbeta_dgamma_kernel(const float* __restrict__ grad_out, const float* __restrict__ x,
                                       const float* __restrict__ mean, const float* __restrict__ invstd,
                                       float* __restrict__ dweight, float* __restrict__ dbias, int N, int C, int H,
                                       int W) {
    int c = blockIdx.x;
    if (c >= C) return;
    int HW = H * W;
    int M = N * HW;

    float sum_db = 0.0f;
    float sum_dg = 0.0f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        int n = i / HW;
        int hw = i - n * HW;
        int off = ((n * C + c) * HW) + hw;
        float go = grad_out[off];
        float xhat = (x[off] - mean[c]) * invstd[c];
        sum_db += go;
        sum_dg += go * xhat;
    }

    __shared__ float shared_db[32];
    __shared__ float shared_dg[32];
    float db = blockReduceSum(sum_db, shared_db);
    float dg = blockReduceSum(sum_dg, shared_dg);
    if (threadIdx.x == 0) {
        dbias[c] = db;
        dweight[c] = dg;
    }
}

__global__ void bn_bwd_dx_kernel(const float* __restrict__ grad_out, const float* __restrict__ x,
                                 const float* __restrict__ weight, const float* __restrict__ mean,
                                 const float* __restrict__ invstd, const float* __restrict__ dweight,
                                 const float* __restrict__ dbias, float* __restrict__ grad_x, int N, int C, int H,
                                 int W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (i >= total) return;
    int HW = H * W;
    int tmp = i / HW;
    int c = tmp % C;
    int M = N * HW;
    float go = grad_out[i];
    float xhat = (x[i] - mean[c]) * invstd[c];
    float dx = ((float)M * go - dbias[c] - xhat * dweight[c]) * (weight[c] * invstd[c]) / (float)M;
    grad_x[i] = dx;
}

__global__ void bn_invstd_kernel(const float* __restrict__ var, float* __restrict__ invstd, int C, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C) return;
    invstd[i] = rsqrtf(var[i] + eps);
}

__global__ void bn_running_update_kernel(float* __restrict__ running_mean, float* __restrict__ running_var,
                                        const float* __restrict__ mean, const float* __restrict__ var, int C,
                                        float momentum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C) return;
    float m = mean[i];
    float v = var[i];
    running_mean[i] = (1.0f - momentum) * running_mean[i] + momentum * m;
    running_var[i] = (1.0f - momentum) * running_var[i] + momentum * v;
}

__global__ void cross_entropy_fwd_kernel(const float* __restrict__ logits, const int64_t* __restrict__ targets,
                                         float* __restrict__ probs, float* __restrict__ loss, int N, int C) {
    __shared__ float shared[32];  // warp partial sums
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    float sample_loss = 0.0f;
    if (n < N) {
        const float* l = logits + n * C;
        float maxv = l[0];
        for (int c = 1; c < C; ++c) maxv = fmaxf(maxv, l[c]);
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) sum += expf(l[c] - maxv);
        float inv = 1.0f / sum;
        int64_t t = targets[n];
        float p_t = 0.0f;
        for (int c = 0; c < C; ++c) {
            float p = expf(l[c] - maxv) * inv;
            probs[n * C + c] = p;
            if (c == (int)t) p_t = p;
        }
        sample_loss = -logf(fmaxf(p_t, 1e-12f));
    }
    float block_loss = blockReduceSum(sample_loss, shared);
    if (threadIdx.x == 0) atomicAdd(loss, block_loss);
}

__global__ void scale_kernel(float* __restrict__ x, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= s;
}

__global__ void cross_entropy_bwd_kernel(const float* __restrict__ probs, const int64_t* __restrict__ targets,
                                         float* __restrict__ grad_logits, const float* __restrict__ grad_out,
                                         int N, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (i >= total) return;
    int n = i / C;
    int c = i - n * C;
    float g = probs[i];
    if (c == (int)targets[n]) g -= 1.0f;
    grad_logits[i] = (grad_out[0] * g) / (float)N;
}

__global__ void sgd_update_kernel(float* __restrict__ param, const float* __restrict__ grad, float* __restrict__ vel,
                                  int n, float lr, float momentum, float weight_decay, bool has_velocity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    if (weight_decay != 0.0f) g += weight_decay * param[i];
    float v = has_velocity ? vel[i] : 0.0f;
    v = momentum * v + g;
    param[i] -= lr * v;
    if (has_velocity) vel[i] = v;
}

__global__ void sgd_update_vec4_kernel(float* __restrict__ param, const float* __restrict__ grad,
                                       float* __restrict__ vel, int n4, float lr, float momentum,
                                       float weight_decay) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int off = idx * 4;
    if (off >= n4) return;

    float4 p = reinterpret_cast<float4*>(param)[idx];
    float4 g = reinterpret_cast<const float4*>(grad)[idx];
    float4 v = reinterpret_cast<float4*>(vel)[idx];

    if (weight_decay != 0.0f) {
        g.x += weight_decay * p.x;
        g.y += weight_decay * p.y;
        g.z += weight_decay * p.z;
        g.w += weight_decay * p.w;
    }

    v.x = momentum * v.x + g.x;
    v.y = momentum * v.y + g.y;
    v.z = momentum * v.z + g.z;
    v.w = momentum * v.w + g.w;

    p.x -= lr * v.x;
    p.y -= lr * v.y;
    p.z -= lr * v.z;
    p.w -= lr * v.w;

    reinterpret_cast<float4*>(param)[idx] = p;
    reinterpret_cast<float4*>(vel)[idx] = v;
}

torch::Tensor conv2d_forward_cudnn(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b, int stride,
                                   int padding) {
    const int N = (int)x.size(0);
    const int Cin = (int)x.size(1);
    const int Hin = (int)x.size(2);
    const int Win = (int)x.size(3);
    const int Cout = (int)w.size(0);
    const int Kh = (int)w.size(2);
    const int Kw = (int)w.size(3);
    const int Hout = (Hin + 2 * padding - Kh) / stride + 1;
    const int Wout = (Win + 2 * padding - Kw) / stride + 1;
    auto y = torch::empty({N, Cout, Hout, Wout}, x.options());

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    cudnnHandle_t handle = get_cudnn_handle(stream);

    CudnnTensorDesc x_desc;
    CudnnTensorDesc y_desc;
    CudnnFilterDesc w_desc;
    CudnnConvDesc conv_desc;
    set_tensor4d_desc(x_desc.desc, N, Cin, Hin, Win);
    set_tensor4d_desc(y_desc.desc, N, Cout, Hout, Wout);
    set_filter4d_desc(w_desc.desc, Cout, Cin, Kh, Kw);
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc.desc, padding, padding, stride, stride, 1, 1,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc.desc, 1));
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc.desc, CUDNN_DEFAULT_MATH));

    int n_out = 0;
    int c_out = 0;
    int h_out = 0;
    int w_out = 0;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc.desc, x_desc.desc, w_desc.desc, &n_out, &c_out,
                                                      &h_out, &w_out));
    TORCH_CHECK(n_out == N && c_out == Cout && h_out == Hout && w_out == Wout,
                "conv2d_forward: cuDNN output shape mismatch");

    cudnnConvolutionFwdAlgoPerf_t perf{};
    int algo_count = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(handle, x_desc.desc, w_desc.desc, conv_desc.desc, y_desc.desc, 1,
                                                       &algo_count, &perf));
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc.desc, w_desc.desc, conv_desc.desc, y_desc.desc,
                                                        perf.algo, &workspace_size));
    auto workspace = workspace_size > 0 ? torch::empty({(int64_t)workspace_size}, x.options().dtype(torch::kUInt8))
                                        : torch::empty({0}, x.options().dtype(torch::kUInt8));
    void* workspace_ptr = workspace_size > 0 ? workspace.data_ptr() : nullptr;

    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(handle, &alpha, x_desc.desc, x.data_ptr<float>(), w_desc.desc,
                                        w.data_ptr<float>(), conv_desc.desc, perf.algo, workspace_ptr, workspace_size,
                                        &beta, y_desc.desc, y.data_ptr<float>()));

    if (b.has_value()) {
        CudnnTensorDesc b_desc;
        set_tensor4d_desc(b_desc.desc, 1, Cout, 1, 1);
        float beta_add = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(handle, &alpha, b_desc.desc, b->data_ptr<float>(), &beta_add, y_desc.desc,
                                   y.data_ptr<float>()));
    }
    return y;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv2d_backward_cudnn(torch::Tensor grad_out, torch::Tensor x,
                                                                              torch::Tensor w, bool need_grad_x,
                                                                              bool need_grad_w, bool need_grad_b,
                                                                              int stride, int padding) {
    const int N = (int)x.size(0);
    const int Cin = (int)x.size(1);
    const int Hin = (int)x.size(2);
    const int Win = (int)x.size(3);
    const int Cout = (int)w.size(0);
    const int Kh = (int)w.size(2);
    const int Kw = (int)w.size(3);
    const int Hout = (int)grad_out.size(2);
    const int Wout = (int)grad_out.size(3);

    auto grad_x = need_grad_x ? torch::empty_like(x) : torch::empty({0}, x.options());
    auto grad_w = need_grad_w ? torch::empty_like(w) : torch::empty({0}, x.options());
    auto grad_b = need_grad_b ? torch::empty({Cout}, x.options()) : torch::empty({0}, x.options());

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    cudnnHandle_t handle = get_cudnn_handle(stream);

    CudnnTensorDesc x_desc;
    CudnnTensorDesc y_desc;
    CudnnFilterDesc w_desc;
    CudnnConvDesc conv_desc;
    set_tensor4d_desc(x_desc.desc, N, Cin, Hin, Win);
    set_tensor4d_desc(y_desc.desc, N, Cout, Hout, Wout);
    set_filter4d_desc(w_desc.desc, Cout, Cin, Kh, Kw);
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc.desc, padding, padding, stride, stride, 1, 1,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc.desc, 1));
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc.desc, CUDNN_DEFAULT_MATH));

    float alpha = 1.0f;
    float beta = 0.0f;
    if (need_grad_x) {
        cudnnConvolutionBwdDataAlgoPerf_t perf{};
        int algo_count = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, w_desc.desc, y_desc.desc, conv_desc.desc,
                                                                x_desc.desc, 1, &algo_count, &perf));
        size_t workspace_size = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, w_desc.desc, y_desc.desc, conv_desc.desc,
                                                                 x_desc.desc, perf.algo, &workspace_size));
        auto workspace =
            workspace_size > 0 ? torch::empty({(int64_t)workspace_size}, x.options().dtype(torch::kUInt8))
                               : torch::empty({0}, x.options().dtype(torch::kUInt8));
        void* workspace_ptr = workspace_size > 0 ? workspace.data_ptr() : nullptr;
        CUDNN_CHECK(cudnnConvolutionBackwardData(handle, &alpha, w_desc.desc, w.data_ptr<float>(), y_desc.desc,
                                                 grad_out.data_ptr<float>(), conv_desc.desc, perf.algo, workspace_ptr,
                                                 workspace_size, &beta, x_desc.desc, grad_x.data_ptr<float>()));
    }

    if (need_grad_w) {
        cudnnConvolutionBwdFilterAlgoPerf_t perf{};
        int algo_count = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, x_desc.desc, y_desc.desc, conv_desc.desc,
                                                                  w_desc.desc, 1, &algo_count, &perf));
        size_t workspace_size = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, x_desc.desc, y_desc.desc, conv_desc.desc,
                                                                   w_desc.desc, perf.algo, &workspace_size));
        auto workspace =
            workspace_size > 0 ? torch::empty({(int64_t)workspace_size}, x.options().dtype(torch::kUInt8))
                               : torch::empty({0}, x.options().dtype(torch::kUInt8));
        void* workspace_ptr = workspace_size > 0 ? workspace.data_ptr() : nullptr;
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle, &alpha, x_desc.desc, x.data_ptr<float>(), y_desc.desc,
                                                   grad_out.data_ptr<float>(), conv_desc.desc, perf.algo,
                                                   workspace_ptr, workspace_size, &beta, w_desc.desc,
                                                   grad_w.data_ptr<float>()));
    }

    if (need_grad_b) {
        CudnnTensorDesc b_desc;
        set_tensor4d_desc(b_desc.desc, 1, Cout, 1, 1);
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle, &alpha, y_desc.desc, grad_out.data_ptr<float>(), &beta,
                                                 b_desc.desc, grad_b.data_ptr<float>()));
    }
    return {grad_x, grad_w, grad_b};
}

}  // namespace

torch::Tensor conv2d_forward(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b, int64_t stride,
                             int64_t padding) {
    check_float_cuda(x, "x");
    check_float_cuda(w, "w");
    if (b.has_value()) check_float_cuda(*b, "b");

    const auto N = (int)x.size(0);
    const auto Cin = (int)x.size(1);
    const auto Hin = (int)x.size(2);
    const auto Win = (int)x.size(3);
    const auto Cout = (int)w.size(0);
    const auto Kh = (int)w.size(2);
    const auto Kw = (int)w.size(3);
    TORCH_CHECK((int)w.size(1) == Cin, "w.shape[1] must match x.shape[1]");
    if (b.has_value()) TORCH_CHECK((int)b->numel() == Cout, "bias numel must equal Cout");

    const int Hout = (Hin + 2 * (int)padding - Kh) / (int)stride + 1;
    const int Wout = (Win + 2 * (int)padding - Kw) / (int)stride + 1;
    TORCH_CHECK(Hout > 0 && Wout > 0, "Invalid output size for conv2d (check stride/padding/kernel)");

    if (cudnn_enabled()) {
        try {
            return conv2d_forward_cudnn(x, w, b, (int)stride, (int)padding);
        } catch (const c10::Error& e) {
            TORCH_WARN("conv2d_forward: cuDNN failed, falling back to custom kernel: ", e.what());
        }
    }

    auto y = torch::empty({N, Cout, Hout, Wout}, x.options());

    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    dim3 block(16, 16, 1);
    const int64_t nc = (int64_t)N * (int64_t)Cout;
    if (nc <= 65535) {
        dim3 grid((Wout + block.x - 1) / block.x, (Hout + block.y - 1) / block.y, (unsigned)nc);
        if (Kh == 3 && Kw == 3 && stride == 1 && padding == 1) {
            conv2d_fwd_3x3s1p1_tiled_kernel<16, 16, 4>
                <<<grid, block, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(),
                                             b.has_value() ? b->data_ptr<float>() : nullptr, y.data_ptr<float>(), N,
                                             Cin, Hin, Win, Cout, Hout, Wout, b.has_value());
        } else {
            conv2d_fwd_kernel<<<grid, block, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(),
                                                          b.has_value() ? b->data_ptr<float>() : nullptr,
                                                          y.data_ptr<float>(), N, Cin, Hin, Win, Cout, Kh, Kw, Hout,
                                                          Wout, (int)stride, (int)padding, b.has_value());
        }
    } else {
        // Avoid grid.z overflow for large batch/channel products by tiling output channels.
        // Note: this path uses the generic kernel (not the 3x3 tiled fast path) for simplicity.
        int oc_tile = 1;
        while ((int64_t)N * (int64_t)((Cout + oc_tile - 1) / oc_tile) > 65535 && oc_tile < 16) oc_tile *= 2;
        const int oc_tiles = (Cout + oc_tile - 1) / oc_tile;
        TORCH_CHECK((int64_t)N * (int64_t)oc_tiles <= 65535, "conv2d_forward: N*Cout too large for kernel launch");
        dim3 grid((Wout + block.x - 1) / block.x, (Hout + block.y - 1) / block.y, (unsigned)(N * oc_tiles));
        const float* bp = b.has_value() ? b->data_ptr<float>() : nullptr;
        if (oc_tile == 16) {
            conv2d_fwd_kernel_oc_tile<16><<<grid, block, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(), bp,
                                                                      y.data_ptr<float>(), N, Cin, Hin, Win, Cout, Kh,
                                                                      Kw, Hout, Wout, (int)stride, (int)padding,
                                                                      b.has_value(), oc_tiles);
        } else if (oc_tile == 8) {
            conv2d_fwd_kernel_oc_tile<8><<<grid, block, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(), bp,
                                                                     y.data_ptr<float>(), N, Cin, Hin, Win, Cout, Kh,
                                                                     Kw, Hout, Wout, (int)stride, (int)padding,
                                                                     b.has_value(), oc_tiles);
        } else if (oc_tile == 4) {
            conv2d_fwd_kernel_oc_tile<4><<<grid, block, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(), bp,
                                                                     y.data_ptr<float>(), N, Cin, Hin, Win, Cout, Kh,
                                                                     Kw, Hout, Wout, (int)stride, (int)padding,
                                                                     b.has_value(), oc_tiles);
        } else if (oc_tile == 2) {
            conv2d_fwd_kernel_oc_tile<2><<<grid, block, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(), bp,
                                                                     y.data_ptr<float>(), N, Cin, Hin, Win, Cout, Kh,
                                                                     Kw, Hout, Wout, (int)stride, (int)padding,
                                                                     b.has_value(), oc_tiles);
        } else {
            conv2d_fwd_kernel_oc_tile<1><<<grid, block, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(), bp,
                                                                     y.data_ptr<float>(), N, Cin, Hin, Win, Cout, Kh,
                                                                     Kw, Hout, Wout, (int)stride, (int)padding,
                                                                     b.has_value(), oc_tiles);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv2d_backward(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor w,
    bool need_grad_x,
    bool need_grad_w,
    bool need_grad_b,
    int64_t stride,
    int64_t padding) {
    check_float_cuda(grad_out, "grad_out");
    check_float_cuda(x, "x");
    check_float_cuda(w, "w");
    const auto N = (int)x.size(0);
    const auto Cin = (int)x.size(1);
    const auto Hin = (int)x.size(2);
    const auto Win = (int)x.size(3);
    const auto Cout = (int)w.size(0);
    const auto Kh = (int)w.size(2);
    const auto Kw = (int)w.size(3);
    const auto Hout = (int)grad_out.size(2);
    const auto Wout = (int)grad_out.size(3);

    if (cudnn_enabled()) {
        try {
            return conv2d_backward_cudnn(grad_out, x, w, need_grad_x, need_grad_w, need_grad_b, (int)stride,
                                         (int)padding);
        } catch (const c10::Error& e) {
            TORCH_WARN("conv2d_backward: cuDNN failed, falling back to custom kernel: ", e.what());
        }
    }

    auto grad_x = need_grad_x ? torch::empty_like(x) : torch::empty({0}, x.options());
    auto grad_w = need_grad_w ? torch::empty_like(w) : torch::empty({0}, x.options());
    auto grad_b = need_grad_b ? torch::empty({Cout}, x.options()) : torch::empty({0}, x.options());

    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    dim3 block(16, 16, 1);
    if (need_grad_x) {
        const int64_t ncx = (int64_t)N * (int64_t)Cin;
        if (ncx <= 65535) {
            dim3 grid_x((Win + block.x - 1) / block.x, (Hin + block.y - 1) / block.y, (unsigned)ncx);
            conv2d_bwd_input_kernel<<<grid_x, block, 0, stream>>>(grad_out.data_ptr<float>(), w.data_ptr<float>(),
                                                                 grad_x.data_ptr<float>(), N, Cin, Hin, Win, Cout, Kh,
                                                                 Kw, Hout, Wout, (int)stride, (int)padding);
        } else {
            int ic_tile = 1;
            while ((int64_t)N * (int64_t)((Cin + ic_tile - 1) / ic_tile) > 65535 && ic_tile < 16) ic_tile *= 2;
            const int ic_tiles = (Cin + ic_tile - 1) / ic_tile;
            TORCH_CHECK((int64_t)N * (int64_t)ic_tiles <= 65535, "conv2d_backward: N*Cin too large for kernel launch");
            dim3 grid_x((Win + block.x - 1) / block.x, (Hin + block.y - 1) / block.y, (unsigned)(N * ic_tiles));
            if (ic_tile == 16) {
                conv2d_bwd_input_kernel_ic_tile<16><<<grid_x, block, 0, stream>>>(
                    grad_out.data_ptr<float>(), w.data_ptr<float>(), grad_x.data_ptr<float>(), N, Cin, Hin, Win, Cout,
                    Kh, Kw, Hout, Wout, (int)stride, (int)padding, ic_tiles);
            } else if (ic_tile == 8) {
                conv2d_bwd_input_kernel_ic_tile<8><<<grid_x, block, 0, stream>>>(
                    grad_out.data_ptr<float>(), w.data_ptr<float>(), grad_x.data_ptr<float>(), N, Cin, Hin, Win, Cout,
                    Kh, Kw, Hout, Wout, (int)stride, (int)padding, ic_tiles);
            } else if (ic_tile == 4) {
                conv2d_bwd_input_kernel_ic_tile<4><<<grid_x, block, 0, stream>>>(
                    grad_out.data_ptr<float>(), w.data_ptr<float>(), grad_x.data_ptr<float>(), N, Cin, Hin, Win, Cout,
                    Kh, Kw, Hout, Wout, (int)stride, (int)padding, ic_tiles);
            } else if (ic_tile == 2) {
                conv2d_bwd_input_kernel_ic_tile<2><<<grid_x, block, 0, stream>>>(
                    grad_out.data_ptr<float>(), w.data_ptr<float>(), grad_x.data_ptr<float>(), N, Cin, Hin, Win, Cout,
                    Kh, Kw, Hout, Wout, (int)stride, (int)padding, ic_tiles);
            } else {
                conv2d_bwd_input_kernel_ic_tile<1><<<grid_x, block, 0, stream>>>(
                    grad_out.data_ptr<float>(), w.data_ptr<float>(), grad_x.data_ptr<float>(), N, Cin, Hin, Win, Cout,
                    Kh, Kw, Hout, Wout, (int)stride, (int)padding, ic_tiles);
            }
        }
    }

    constexpr int THREADS = 256;
    if (need_grad_w) {
        dim3 grid_w(Cout, Cin, Kh * Kw);
        conv2d_bwd_weight_kernel<THREADS><<<grid_w, THREADS, 0, stream>>>(
            grad_out.data_ptr<float>(), x.data_ptr<float>(), grad_w.data_ptr<float>(), N, Cin, Hin, Win, Cout, Kh, Kw,
            Hout, Wout, (int)stride, (int)padding);
    }

    if (need_grad_b) {
        conv2d_bwd_bias_kernel<THREADS><<<Cout, THREADS, 0, stream>>>(grad_out.data_ptr<float>(),
                                                                      grad_b.data_ptr<float>(), N, Cout, Hout, Wout);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_x, grad_w, grad_b};
}

torch::Tensor relu_forward(torch::Tensor x) {
    check_float_cuda(x, "x");
    auto y = torch::empty_like(x);
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    int n = (int)x.numel();
    int threads = 256;
    const auto xp = (std::uintptr_t)x.data_ptr<float>();
    const auto yp = (std::uintptr_t)y.data_ptr<float>();
    if ((xp % alignof(float4) == 0) && (yp % alignof(float4) == 0)) {
        int n4 = (n / 4) * 4;
        int blocks4 = ((n4 / 4) + threads - 1) / threads;
        if (n4 > 0) relu_fwd_vec4_kernel<<<blocks4, threads, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(), n4);
        int tail = n - n4;
        if (tail > 0) {
            int blocks = (tail + threads - 1) / threads;
            relu_fwd_kernel<<<blocks, threads, 0, stream>>>(x.data_ptr<float>() + n4, y.data_ptr<float>() + n4, tail);
        }
    } else {
        int blocks = (n + threads - 1) / threads;
        relu_fwd_kernel<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor relu_backward(torch::Tensor grad_out, torch::Tensor x) {
    check_float_cuda(grad_out, "grad_out");
    check_float_cuda(x, "x");
    auto grad_x = torch::empty_like(x);
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    int n = (int)x.numel();
    int threads = 256;
    const auto gop = (std::uintptr_t)grad_out.data_ptr<float>();
    const auto xp = (std::uintptr_t)x.data_ptr<float>();
    const auto gxp = (std::uintptr_t)grad_x.data_ptr<float>();
    if ((gop % alignof(float4) == 0) && (xp % alignof(float4) == 0) && (gxp % alignof(float4) == 0)) {
        int n4 = (n / 4) * 4;
        int blocks4 = ((n4 / 4) + threads - 1) / threads;
        if (n4 > 0) {
            relu_bwd_vec4_kernel<<<blocks4, threads, 0, stream>>>(grad_out.data_ptr<float>(), x.data_ptr<float>(),
                                                                 grad_x.data_ptr<float>(), n4);
        }
        int tail = n - n4;
        if (tail > 0) {
            int blocks = (tail + threads - 1) / threads;
            relu_bwd_kernel<<<blocks, threads, 0, stream>>>(grad_out.data_ptr<float>() + n4, x.data_ptr<float>() + n4,
                                                           grad_x.data_ptr<float>() + n4, tail);
        }
    } else {
        int blocks = (n + threads - 1) / threads;
        relu_bwd_kernel<<<blocks, threads, 0, stream>>>(grad_out.data_ptr<float>(), x.data_ptr<float>(),
                                                       grad_x.data_ptr<float>(), n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x;
}

std::tuple<torch::Tensor, torch::Tensor> maxpool2d_forward(torch::Tensor x, int64_t kernel, int64_t stride) {
    check_float_cuda(x, "x");
    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int Hin = (int)x.size(2);
    const int Win = (int)x.size(3);
    const int Hout = (Hin - (int)kernel) / (int)stride + 1;
    const int Wout = (Win - (int)kernel) / (int)stride + 1;
    TORCH_CHECK(Hout > 0 && Wout > 0, "Invalid output size for maxpool2d (check kernel/stride)");
    auto y = torch::empty({N, C, Hout, Wout}, x.options());
    auto idx = torch::empty({N, C, Hout, Wout}, x.options().dtype(torch::kInt32));

    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    dim3 block(16, 16, 1);
    const int64_t nc = (int64_t)N * (int64_t)C;
    if (nc <= 65535) {
        dim3 grid((Wout + block.x - 1) / block.x, (Hout + block.y - 1) / block.y, (unsigned)nc);
        maxpool2d_fwd_kernel<<<grid, block, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(),
                                                         idx.data_ptr<int32_t>(), N, C, Hin, Win, Hout, Wout,
                                                         (int)kernel, (int)stride);
    } else {
        int c_tile = 1;
        while ((int64_t)N * (int64_t)((C + c_tile - 1) / c_tile) > 65535 && c_tile < 16) c_tile *= 2;
        const int c_tiles = (C + c_tile - 1) / c_tile;
        TORCH_CHECK((int64_t)N * (int64_t)c_tiles <= 65535, "maxpool2d_forward: N*C too large for kernel launch");
        dim3 grid((Wout + block.x - 1) / block.x, (Hout + block.y - 1) / block.y, (unsigned)(N * c_tiles));
        if (c_tile == 16) {
            maxpool2d_fwd_kernel_c_tile<16><<<grid, block, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(),
                                                                        idx.data_ptr<int32_t>(), N, C, Hin, Win, Hout,
                                                                        Wout, (int)kernel, (int)stride, c_tiles);
        } else if (c_tile == 8) {
            maxpool2d_fwd_kernel_c_tile<8><<<grid, block, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(),
                                                                       idx.data_ptr<int32_t>(), N, C, Hin, Win, Hout,
                                                                       Wout, (int)kernel, (int)stride, c_tiles);
        } else if (c_tile == 4) {
            maxpool2d_fwd_kernel_c_tile<4><<<grid, block, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(),
                                                                       idx.data_ptr<int32_t>(), N, C, Hin, Win, Hout,
                                                                       Wout, (int)kernel, (int)stride, c_tiles);
        } else if (c_tile == 2) {
            maxpool2d_fwd_kernel_c_tile<2><<<grid, block, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(),
                                                                       idx.data_ptr<int32_t>(), N, C, Hin, Win, Hout,
                                                                       Wout, (int)kernel, (int)stride, c_tiles);
        } else {
            maxpool2d_fwd_kernel_c_tile<1><<<grid, block, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(),
                                                                       idx.data_ptr<int32_t>(), N, C, Hin, Win, Hout,
                                                                       Wout, (int)kernel, (int)stride, c_tiles);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, idx};
}

torch::Tensor maxpool2d_backward(torch::Tensor grad_out, torch::Tensor indices, int64_t in_h, int64_t in_w,
                                 int64_t kernel, int64_t stride) {
    check_float_cuda(grad_out, "grad_out");
    check_cuda(indices, "indices");
    TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
    const int N = (int)grad_out.size(0);
    const int C = (int)grad_out.size(1);
    const int Hout = (int)grad_out.size(2);
    const int Wout = (int)grad_out.size(3);
    auto grad_x = torch::zeros({N, C, (int)in_h, (int)in_w}, grad_out.options());

    c10::cuda::CUDAGuard guard(grad_out.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    dim3 block(16, 16, 1);
    const int64_t nc = (int64_t)N * (int64_t)C;
    if (nc <= 65535) {
        dim3 grid((Wout + block.x - 1) / block.x, (Hout + block.y - 1) / block.y, (unsigned)nc);
        maxpool2d_bwd_kernel<<<grid, block, 0, stream>>>(grad_out.data_ptr<float>(), indices.data_ptr<int32_t>(),
                                                         grad_x.data_ptr<float>(), N, C, (int)in_h, (int)in_w, Hout,
                                                         Wout);
    } else {
        int c_tile = 1;
        while ((int64_t)N * (int64_t)((C + c_tile - 1) / c_tile) > 65535 && c_tile < 16) c_tile *= 2;
        const int c_tiles = (C + c_tile - 1) / c_tile;
        TORCH_CHECK((int64_t)N * (int64_t)c_tiles <= 65535, "maxpool2d_backward: N*C too large for kernel launch");
        dim3 grid((Wout + block.x - 1) / block.x, (Hout + block.y - 1) / block.y, (unsigned)(N * c_tiles));
        if (c_tile == 16) {
            maxpool2d_bwd_kernel_c_tile<16><<<grid, block, 0, stream>>>(
                grad_out.data_ptr<float>(), indices.data_ptr<int32_t>(), grad_x.data_ptr<float>(), N, C, (int)in_h,
                (int)in_w, Hout, Wout, c_tiles);
        } else if (c_tile == 8) {
            maxpool2d_bwd_kernel_c_tile<8><<<grid, block, 0, stream>>>(
                grad_out.data_ptr<float>(), indices.data_ptr<int32_t>(), grad_x.data_ptr<float>(), N, C, (int)in_h,
                (int)in_w, Hout, Wout, c_tiles);
        } else if (c_tile == 4) {
            maxpool2d_bwd_kernel_c_tile<4><<<grid, block, 0, stream>>>(
                grad_out.data_ptr<float>(), indices.data_ptr<int32_t>(), grad_x.data_ptr<float>(), N, C, (int)in_h,
                (int)in_w, Hout, Wout, c_tiles);
        } else if (c_tile == 2) {
            maxpool2d_bwd_kernel_c_tile<2><<<grid, block, 0, stream>>>(
                grad_out.data_ptr<float>(), indices.data_ptr<int32_t>(), grad_x.data_ptr<float>(), N, C, (int)in_h,
                (int)in_w, Hout, Wout, c_tiles);
        } else {
            maxpool2d_bwd_kernel_c_tile<1><<<grid, block, 0, stream>>>(
                grad_out.data_ptr<float>(), indices.data_ptr<int32_t>(), grad_x.data_ptr<float>(), N, C, (int)in_h,
                (int)in_w, Hout, Wout, c_tiles);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x;
}

torch::Tensor global_avg_pool2d_forward(torch::Tensor x) {
    check_float_cuda(x, "x");
    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);
    auto y = torch::empty({N, C}, x.options());
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    int threads = 256;
    dim3 block(threads, 1, 1);
    dim3 grid((C + threads - 1) / threads, N, 1);
    global_avg_pool_fwd_kernel<<<grid, block, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(), N, C, H, W);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor global_avg_pool2d_backward(torch::Tensor grad_out, int64_t h, int64_t w) {
    check_float_cuda(grad_out, "grad_out");
    const int N = (int)grad_out.size(0);
    const int C = (int)grad_out.size(1);
    auto grad_x = torch::empty({N, C, (int)h, (int)w}, grad_out.options());
    c10::cuda::CUDAGuard guard(grad_out.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    int total = N * C * (int)h * (int)w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    global_avg_pool_bwd_kernel<<<blocks, threads, 0, stream>>>(grad_out.data_ptr<float>(), grad_x.data_ptr<float>(),
                                                              N, C, (int)h, (int)w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x;
}

torch::Tensor linear_forward(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b) {
    check_float_cuda(x, "x");
    check_float_cuda(w, "w");
    if (b.has_value()) check_float_cuda(*b, "b");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (N, In)");
    TORCH_CHECK(w.dim() == 2, "w must be 2D (Out, In)");
    const int N = (int)x.size(0);
    const int In = (int)x.size(1);
    const int Out = (int)w.size(0);
    TORCH_CHECK((int)w.size(1) == In, "w.shape[1] must match x.shape[1]");
    if (b.has_value()) TORCH_CHECK((int)b->numel() == Out, "bias numel must equal Out");
    auto y = torch::empty({N, Out}, x.options());
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    constexpr int TILE = 16;
    dim3 block(TILE, TILE, 1);
    dim3 grid((Out + TILE - 1) / TILE, (N + TILE - 1) / TILE, 1);
    linear_fwd_tiled_kernel<TILE><<<grid, block, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(),
                                                             b.has_value() ? b->data_ptr<float>() : nullptr,
                                                             y.data_ptr<float>(), N, In, Out, b.has_value());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> linear_backward(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor w,
    bool need_grad_x,
    bool need_grad_w,
    bool need_grad_b) {
    check_float_cuda(grad_out, "grad_out");
    check_float_cuda(x, "x");
    check_float_cuda(w, "w");
    const int N = (int)x.size(0);
    const int In = (int)x.size(1);
    const int Out = (int)w.size(0);
    auto grad_x = need_grad_x ? torch::empty_like(x) : torch::empty({0}, x.options());
    auto grad_w = need_grad_w ? torch::empty_like(w) : torch::empty({0}, x.options());
    auto grad_b = need_grad_b ? torch::empty({Out}, x.options()) : torch::empty({0}, x.options());

    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    int threads = 256;
    dim3 block(threads, 1, 1);
    if (need_grad_x) {
        dim3 grid_x((In + threads - 1) / threads, N, 1);
        linear_bwd_input_kernel<<<grid_x, block, 0, stream>>>(grad_out.data_ptr<float>(), w.data_ptr<float>(),
                                                             grad_x.data_ptr<float>(), N, In, Out);
    }

    constexpr int RTHREADS = 256;
    if (need_grad_w) {
        dim3 grid_w(Out, In, 1);
        linear_bwd_weight_kernel<RTHREADS><<<grid_w, RTHREADS, 0, stream>>>(
            grad_out.data_ptr<float>(), x.data_ptr<float>(), grad_w.data_ptr<float>(), N, In, Out);
    }

    if (need_grad_b) {
        linear_bwd_bias_kernel<RTHREADS><<<Out, RTHREADS, 0, stream>>>(grad_out.data_ptr<float>(),
                                                                       grad_b.data_ptr<float>(), N, Out);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_x, grad_w, grad_b};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> batchnorm2d_forward(torch::Tensor x, torch::Tensor weight,
                                                                            torch::Tensor bias, torch::Tensor running_mean,
                                                                            torch::Tensor running_var, bool training,
                                                                            double momentum, double eps) {
    check_float_cuda(x, "x");
    check_float_cuda(weight, "weight");
    check_float_cuda(bias, "bias");
    check_float_cuda(running_mean, "running_mean");
    check_float_cuda(running_var, "running_var");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");
    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);
    TORCH_CHECK(weight.numel() == C && bias.numel() == C, "weight/bias must have shape (C)");
    TORCH_CHECK(running_mean.numel() == C && running_var.numel() == C, "running stats must have shape (C)");

    auto y = torch::empty_like(x);
    torch::Tensor mean = training ? torch::empty({C}, x.options()) : running_mean;
    auto invstd = torch::empty({C}, x.options());

    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (training) {
        auto var = torch::empty({C}, x.options());
        int threads = 256;
        bn_stats_kernel<<<C, threads, 0, stream>>>(x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(),
                                                   N, C, H, W);
        int blocks = (C + threads - 1) / threads;
        // invstd = rsqrt(var + eps)
        // reuse scale_kernel (x[i] *= s) style would not fit; do a tiny kernel inline by using scale + add isn't available.
        // Here compute invstd with a simple elementwise kernel via ATen? Not allowed. Implement a custom kernel below.
        // For simplicity: launch bn_fwd_kernel expects invstd; compute invstd in a small CUDA kernel here.
        // We use a lambda-free kernel by reusing global_avg_pool_bwd style isn't possible; define below.
        // (Implemented as bn_invstd_kernel.)
        // NOLINTNEXTLINE(bugprone-use-after-move)
        bn_invstd_kernel<<<blocks, threads, 0, stream>>>(var.data_ptr<float>(), invstd.data_ptr<float>(), C, (float)eps);

        // Update running stats (in-place)
        bn_running_update_kernel<<<blocks, threads, 0, stream>>>(running_mean.data_ptr<float>(),
                                                                 running_var.data_ptr<float>(), mean.data_ptr<float>(),
                                                                 var.data_ptr<float>(), C, (float)momentum);
    } else {
        int threads = 256;
        int blocks = (C + threads - 1) / threads;
        bn_invstd_kernel<<<blocks, threads, 0, stream>>>(running_var.data_ptr<float>(), invstd.data_ptr<float>(), C,
                                                         (float)eps);
    }

    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    bn_fwd_kernel<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                                                  mean.data_ptr<float>(), invstd.data_ptr<float>(), y.data_ptr<float>(),
                                                  N, C, H, W);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, mean, invstd};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> batchnorm2d_backward(torch::Tensor grad_out, torch::Tensor x,
                                                                             torch::Tensor weight, torch::Tensor saved_mean,
                                                                             torch::Tensor saved_invstd, bool need_grad_x,
                                                                             bool need_grad_w, bool need_grad_b) {
    check_float_cuda(grad_out, "grad_out");
    check_float_cuda(x, "x");
    check_float_cuda(weight, "weight");
    check_float_cuda(saved_mean, "saved_mean");
    check_float_cuda(saved_invstd, "saved_invstd");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");
    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);

    auto grad_x = need_grad_x ? torch::empty_like(x) : torch::empty({0}, x.options());
    auto grad_w = need_grad_w ? torch::empty({C}, x.options()) : torch::empty({0}, x.options());
    auto grad_b = need_grad_b ? torch::empty({C}, x.options()) : torch::empty({0}, x.options());

    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    int threads = 256;
    torch::Tensor tmp_dw;
    torch::Tensor tmp_db;
    if (need_grad_w || need_grad_b || need_grad_x) {
        // For dx we need dgamma and dbeta; if caller doesn't request them but dx does, compute into temporaries.
        tmp_dw = need_grad_w ? grad_w : torch::empty({C}, x.options());
        tmp_db = need_grad_b ? grad_b : torch::empty({C}, x.options());

        bn_dbeta_dgamma_kernel<<<C, threads, 0, stream>>>(grad_out.data_ptr<float>(), x.data_ptr<float>(),
                                                          saved_mean.data_ptr<float>(), saved_invstd.data_ptr<float>(),
                                                          tmp_dw.data_ptr<float>(), tmp_db.data_ptr<float>(), N, C, H, W);
    }

    if (need_grad_x) {
        int total = N * C * H * W;
        TORCH_CHECK(tmp_dw.numel() == C && tmp_db.numel() == C, "batchnorm2d_backward internal error: missing stats");
        int blocks = (total + threads - 1) / threads;
        bn_bwd_dx_kernel<<<blocks, threads, 0, stream>>>(grad_out.data_ptr<float>(), x.data_ptr<float>(),
                                                         weight.data_ptr<float>(), saved_mean.data_ptr<float>(),
                                                         saved_invstd.data_ptr<float>(), tmp_dw.data_ptr<float>(),
                                                         tmp_db.data_ptr<float>(), grad_x.data_ptr<float>(), N, C, H, W);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_x, grad_w, grad_b};
}

std::tuple<torch::Tensor, torch::Tensor> cross_entropy_forward(torch::Tensor logits, torch::Tensor targets) {
    check_float_cuda(logits, "logits");
    check_cuda(targets, "targets");
    TORCH_CHECK(targets.scalar_type() == at::kLong, "targets must be int64");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D (N, C)");
    TORCH_CHECK(targets.dim() == 1, "targets must be 1D (N)");
    const int N = (int)logits.size(0);
    const int C = (int)logits.size(1);
    TORCH_CHECK((int)targets.size(0) == N, "targets.size(0) must match logits.size(0)");
    auto probs = torch::empty_like(logits);
    auto loss = torch::zeros({}, logits.options());

    c10::cuda::CUDAGuard guard(logits.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cross_entropy_fwd_kernel<<<blocks, threads, 0, stream>>>(logits.data_ptr<float>(), targets.data_ptr<int64_t>(),
                                                            probs.data_ptr<float>(), loss.data_ptr<float>(), N, C);
    // loss = loss / N
    float scale = 1.0f / (float)N;
    scale_kernel<<<1, 1, 0, stream>>>(loss.data_ptr<float>(), 1, scale);
    return {loss, probs};
}

torch::Tensor cross_entropy_backward(torch::Tensor probs, torch::Tensor targets, torch::Tensor grad_out) {
    check_float_cuda(probs, "probs");
    check_cuda(targets, "targets");
    check_float_cuda(grad_out, "grad_out");
    TORCH_CHECK(targets.scalar_type() == at::kLong, "targets must be int64");
    TORCH_CHECK(grad_out.numel() == 1, "grad_out must be scalar");
    TORCH_CHECK(probs.dim() == 2, "probs must be 2D (N, C)");
    const int N = (int)probs.size(0);
    const int C = (int)probs.size(1);
    auto grad_logits = torch::empty_like(probs);

    c10::cuda::CUDAGuard guard(probs.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    int total = N * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cross_entropy_bwd_kernel<<<blocks, threads, 0, stream>>>(probs.data_ptr<float>(), targets.data_ptr<int64_t>(),
                                                            grad_logits.data_ptr<float>(), grad_out.data_ptr<float>(),
                                                            N, C);
    return grad_logits;
}

void sgd_update_(torch::Tensor param, torch::Tensor grad, c10::optional<torch::Tensor> velocity, double lr,
                 double momentum, double weight_decay) {
    check_float_cuda(param, "param");
    check_float_cuda(grad, "grad");
    TORCH_CHECK(param.numel() == grad.numel(), "param/grad must have same numel");
    if (velocity.has_value()) {
        check_float_cuda(*velocity, "velocity");
        TORCH_CHECK(velocity->numel() == param.numel(), "velocity must match param");
    }
    c10::cuda::CUDAGuard guard(param.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    int n = (int)param.numel();
    int threads = 256;
    if (velocity.has_value() && momentum != 0.0) {
        const auto pp = (std::uintptr_t)param.data_ptr<float>();
        const auto gp = (std::uintptr_t)grad.data_ptr<float>();
        const auto vp = (std::uintptr_t)velocity->data_ptr<float>();
        if ((pp % alignof(float4) == 0) && (gp % alignof(float4) == 0) && (vp % alignof(float4) == 0)) {
            int n4 = (n / 4) * 4;
            int blocks4 = ((n4 / 4) + threads - 1) / threads;
            if (n4 > 0) {
                sgd_update_vec4_kernel<<<blocks4, threads, 0, stream>>>(
                    param.data_ptr<float>(), grad.data_ptr<float>(), velocity->data_ptr<float>(), n4, (float)lr,
                    (float)momentum, (float)weight_decay);
            }
            int tail = n - n4;
            if (tail > 0) {
                int blocks = (tail + threads - 1) / threads;
                sgd_update_kernel<<<blocks, threads, 0, stream>>>(param.data_ptr<float>() + n4, grad.data_ptr<float>() + n4,
                                                                 velocity->data_ptr<float>() + n4, tail, (float)lr,
                                                                 (float)momentum, (float)weight_decay, true);
            }
            return;
        }
    }
    int blocks = (n + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads, 0, stream>>>(param.data_ptr<float>(), grad.data_ptr<float>(),
                                                     velocity.has_value() ? velocity->data_ptr<float>() : nullptr, n,
                                                     (float)lr, (float)momentum, (float)weight_decay,
                                                     velocity.has_value());
}
