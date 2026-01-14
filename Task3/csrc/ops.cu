#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>
#include <tuple>

namespace {

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
    float v = x[i];
    y[i] = v > 0.0f ? v : 0.0f;
}

__global__ void relu_bwd_kernel(const float* __restrict__ grad_out, const float* __restrict__ x,
                                float* __restrict__ grad_x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    grad_x[i] = x[i] > 0.0f ? grad_out[i] : 0.0f;
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

__global__ void cross_entropy_fwd_kernel(const float* __restrict__ logits, const int64_t* __restrict__ targets,
                                         float* __restrict__ probs, float* __restrict__ loss, int N, int C) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
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
    float sample_loss = -logf(fmaxf(p_t, 1e-12f));
    atomicAdd(loss, sample_loss);
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
    auto y = torch::empty({N, Cout, Hout, Wout}, x.options());

    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    dim3 block(16, 16, 1);
    dim3 grid((Wout + block.x - 1) / block.x, (Hout + block.y - 1) / block.y, N * Cout);
    conv2d_fwd_kernel<<<grid, block, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(),
                                                  b.has_value() ? b->data_ptr<float>() : nullptr, y.data_ptr<float>(),
                                                  N, Cin, Hin, Win, Cout, Kh, Kw, Hout, Wout, (int)stride,
                                                  (int)padding, b.has_value());
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

    auto grad_x = need_grad_x ? torch::empty_like(x) : torch::empty({0}, x.options());
    auto grad_w = need_grad_w ? torch::empty_like(w) : torch::empty({0}, x.options());
    auto grad_b = need_grad_b ? torch::empty({Cout}, x.options()) : torch::empty({0}, x.options());

    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    dim3 block(16, 16, 1);
    if (need_grad_x) {
        dim3 grid_x((Win + block.x - 1) / block.x, (Hin + block.y - 1) / block.y, N * Cin);
        conv2d_bwd_input_kernel<<<grid_x, block, 0, stream>>>(grad_out.data_ptr<float>(), w.data_ptr<float>(),
                                                             grad_x.data_ptr<float>(), N, Cin, Hin, Win, Cout, Kh, Kw,
                                                             Hout, Wout, (int)stride, (int)padding);
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
    return {grad_x, grad_w, grad_b};
}

torch::Tensor relu_forward(torch::Tensor x) {
    check_float_cuda(x, "x");
    auto y = torch::empty_like(x);
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    int n = (int)x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_fwd_kernel<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
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
    int blocks = (n + threads - 1) / threads;
    relu_bwd_kernel<<<blocks, threads, 0, stream>>>(grad_out.data_ptr<float>(), x.data_ptr<float>(),
                                                   grad_x.data_ptr<float>(), n);
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
    auto y = torch::empty({N, C, Hout, Wout}, x.options());
    auto idx = torch::empty({N, C, Hout, Wout}, x.options().dtype(torch::kInt32));

    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    dim3 block(16, 16, 1);
    dim3 grid((Wout + block.x - 1) / block.x, (Hout + block.y - 1) / block.y, N * C);
    maxpool2d_fwd_kernel<<<grid, block, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(),
                                                     idx.data_ptr<int32_t>(), N, C, Hin, Win, Hout, Wout,
                                                     (int)kernel, (int)stride);
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
    dim3 grid((Wout + block.x - 1) / block.x, (Hout + block.y - 1) / block.y, N * C);
    maxpool2d_bwd_kernel<<<grid, block, 0, stream>>>(grad_out.data_ptr<float>(), indices.data_ptr<int32_t>(),
                                                     grad_x.data_ptr<float>(), N, C, (int)in_h, (int)in_w, Hout,
                                                     Wout);
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
    int threads = 256;
    dim3 block(threads, 1, 1);
    dim3 grid((Out + threads - 1) / threads, N, 1);
    linear_fwd_kernel<<<grid, block, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(),
                                                 b.has_value() ? b->data_ptr<float>() : nullptr, y.data_ptr<float>(),
                                                 N, In, Out, b.has_value());
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
    int blocks = (n + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads, 0, stream>>>(param.data_ptr<float>(), grad.data_ptr<float>(),
                                                     velocity.has_value() ? velocity->data_ptr<float>() : nullptr, n,
                                                     (float)lr, (float)momentum, (float)weight_decay,
                                                     velocity.has_value());
}
