#include "cuda_fill.cuh"

#include <cmath>
#include <iostream>
#include <vector>

static bool nearly_equal(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

static bool run_fill_case(size_t n, float value) {
    if (n == 0) {
        return true;
    }

    float* d_data = nullptr;
    if (cudaMalloc(&d_data, n * sizeof(float)) != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        return false;
    }

    fill_device(d_data, n, value);
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(launch_err) << "\n";
        cudaFree(d_data);
        return false;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed\n";
        cudaFree(d_data);
        return false;
    }

    std::vector<float> h(n, 0.0f);
    if (cudaMemcpy(h.data(), d_data, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "cudaMemcpy failed\n";
        cudaFree(d_data);
        return false;
    }
    cudaFree(d_data);

    for (size_t i = 0; i < n; ++i) {
        if (!nearly_equal(h[i], value)) {
            std::cerr << "Mismatch at index " << i << ": expected " << value << ", got " << h[i]
                      << "\n";
            return false;
        }
    }
    return true;
}

int main() {
    struct Case {
        size_t n;
        float value;
    };

    Case cases[] = {
        {1, 1.0f},
        {255, 1.0f},
        {256, 1.0f},
        {257, 1.0f},
        {1027, -3.5f},
        {4096, 0.0f},
    };

    for (const auto& c : cases) {
        if (!run_fill_case(c.n, c.value)) {
            std::cerr << "fill_device test failed (n=" << c.n << ", value=" << c.value << ")\n";
            return 1;
        }
    }

    std::cout << "fill_device tests passed!\n";
    return 0;
}

