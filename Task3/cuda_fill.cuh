#pragma once

#include <cuda_runtime.h>
#include <cstddef>

__global__ void fill_kernel(float* data, size_t count, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    data[idx] = value;
}

inline void fill_device(float* data, size_t count, float value) {
    if (count == 0) {
        return;
    }
    int threads = 256;
    int blocks = static_cast<int>((count + threads - 1) / threads);
    fill_kernel<<<blocks, threads>>>(data, count, value);
}

