#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuda_fill.cuh"
#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void scale_shift_kernel(float* data, size_t count, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    float val = data[idx];
    data[idx] = (val - 0.5f) * 2.0f * scale;
}

__global__ void im2col_kernel(const int n, const float* data_im,
                              int batch, int channels, int height, int width,
                              int kernel_h, int kernel_w,
                              int pad_h, int pad_w,
                              int stride_h, int stride_w,
                              int height_col, int width_col,
                              float* data_col) {
    CUDA_KERNEL_LOOP(index, n) {
        int w_out = index % width_col;
        int h_out = (index / width_col) % height_col;
        int k_w = (index / (width_col * height_col)) % kernel_w;
        int k_h = (index / (width_col * height_col * kernel_w)) % kernel_h;
        int channel = (index / (width_col * height_col * kernel_w * kernel_h)) % channels;
        int batch_idx = index / (width_col * height_col * kernel_w * kernel_h * channels);

        int h_in = h_out * stride_h - pad_h + k_h;
        int w_in = w_out * stride_w - pad_w + k_w;

        int data_col_index = index;
        if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
            int data_im_index = (((batch_idx * channels + channel) * height + h_in) * width) + w_in;
            data_col[data_col_index] = data_im[data_im_index];
        } else {
            data_col[data_col_index] = 0.f;
        }
    }
}

__global__ void col2im_kernel(const int n, const float* data_col,
                              int batch, int channels, int height, int width,
                              int kernel_h, int kernel_w,
                              int pad_h, int pad_w,
                              int stride_h, int stride_w,
                              int height_col, int width_col,
                              float* data_im) {
    CUDA_KERNEL_LOOP(index, n) {
        int w = index % width;
        int h = (index / width) % height;
        int channel = (index / (width * height)) % channels;
        int batch_idx = index / (channels * height * width);

        int c_offset = channel * kernel_h * kernel_w;
        float val = 0.f;
        for (int k_h = 0; k_h < kernel_h; ++k_h) {
            int h_col = h + pad_h - k_h;
            if (h_col < 0 || h_col % stride_h != 0) {
                continue;
            }
            h_col /= stride_h;
            if (h_col < 0 || h_col >= height_col) {
                continue;
            }
            for (int k_w = 0; k_w < kernel_w; ++k_w) {
                int w_col = w + pad_w - k_w;
                if (w_col < 0 || w_col % stride_w != 0) {
                    continue;
                }
                w_col /= stride_w;
                if (w_col < 0 || w_col >= width_col) {
                    continue;
                }
                int col_index =
                    (((batch_idx * (channels * kernel_h * kernel_w) +
                       (c_offset + k_h * kernel_w + k_w)) *
                      height_col + h_col) *
                     width_col + w_col);
                val += data_col[col_index];
            }
        }
        data_im[index] = val;
    }
}

void im2col_gpu(const float* data_im, int batch, int channels, int height, int width,
                int kernel_h, int kernel_w, int pad_h, int pad_w,
                int stride_h, int stride_w, int height_col, int width_col,
                float* data_col) {
    int channels_col = channels * kernel_h * kernel_w;
    int n = batch * channels_col * height_col * width_col;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    im2col_kernel<<<blocks, threads>>>(n, data_im, batch, channels, height, width,
                                       kernel_h, kernel_w, pad_h, pad_w,
                                       stride_h, stride_w, height_col, width_col,
                                       data_col);
}

void col2im_gpu(const float* data_col, int batch, int channels, int height, int width,
                int kernel_h, int kernel_w, int pad_h, int pad_w,
                int stride_h, int stride_w, int height_col, int width_col,
                float* data_im) {
    int n = batch * channels * height * width;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    col2im_kernel<<<blocks, threads>>>(n, data_col, batch, channels, height, width,
                                       kernel_h, kernel_w, pad_h, pad_w,
                                       stride_h, stride_w, height_col, width_col,
                                       data_im);
}

__global__ void matrix_to_nchw_kernel(const float* matrix, float* tensor,
                                      int batch, int channels, int height, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial = height * width;
    int total = batch * channels * spatial;
    if (index >= total) {
        return;
    }
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / (width * height)) % channels;
    int n = index / (channels * height * width);

    int col = n * spatial + h * width + w;
    tensor[index] = matrix[c * (batch * spatial) + col];
}

__global__ void nchw_to_matrix_kernel(const float* tensor, float* matrix,
                                      int batch, int channels, int height, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial = height * width;
    int total = batch * channels * spatial;
    if (index >= total) {
        return;
    }
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / (width * height)) % channels;
    int n = index / (channels * height * width);
    int col = n * spatial + h * width + w;
    matrix[c * (batch * spatial) + col] = tensor[index];
}

void matrix_to_nchw(const float* matrix, float* tensor,
                    int batch, int channels, int height, int width) {
    int total = batch * channels * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    matrix_to_nchw_kernel<<<blocks, threads>>>(matrix, tensor, batch, channels, height, width);
}

void nchw_to_matrix(const float* tensor, float* matrix,
                    int batch, int channels, int height, int width) {
    int total = batch * channels * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    nchw_to_matrix_kernel<<<blocks, threads>>>(tensor, matrix, batch, channels, height, width);
}

void conv2d_forward(const float* input, const float* weights, const float* bias,
                    float* output, int batch, int in_channels, int out_channels,
                    int height, int width, int kernel_h = 3, int kernel_w = 3,
                    int pad_h = 1, int pad_w = 1, int stride_h = 1, int stride_w = 1) {
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int kernel_dim = in_channels * kernel_h * kernel_w;
    int n_cols = batch * height_col * width_col;

    float* col_buffer;
    cudaMalloc(&col_buffer, kernel_dim * n_cols * sizeof(float));
    im2col_gpu(input, batch, in_channels, height, width,
               kernel_h, kernel_w, pad_h, pad_w,
               stride_h, stride_w, height_col, width_col,
               col_buffer);

    float* output_matrix;
    cudaMalloc(&output_matrix, out_channels * n_cols * sizeof(float));
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n_cols, out_channels, kernel_dim,
                &alpha,
                col_buffer, n_cols,
                weights, kernel_dim,
                &beta,
                output_matrix, n_cols);

    if (bias != nullptr) {
        float* ones;
        cudaMalloc(&ones, n_cols * sizeof(float));
        fill_device(ones, static_cast<size_t>(n_cols), 1.0f);
        const float alpha_bias = 1.0f;
        const float beta_bias = 1.0f;
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n_cols, out_channels, 1,
                    &alpha_bias,
                    ones, n_cols,
                    bias, 1,
                    &beta_bias,
                    output_matrix, n_cols);
        cudaFree(ones);
    }

    matrix_to_nchw(output_matrix, output, batch, out_channels, height_col, width_col);

    cublasDestroy(handle);
    cudaFree(output_matrix);
    cudaFree(col_buffer);
}

void conv2d_backward(const float* input, const float* grad_output, const float* weights,
                     float* grad_input, float* grad_weights, float* grad_bias,
                     int batch, int in_channels, int out_channels,
                     int height, int width, int kernel_h = 3, int kernel_w = 3,
                     int pad_h = 1, int pad_w = 1, int stride_h = 1, int stride_w = 1) {
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int kernel_dim = in_channels * kernel_h * kernel_w;
    int n_cols = batch * height_col * width_col;

    float* col_buffer;
    cudaMalloc(&col_buffer, kernel_dim * n_cols * sizeof(float));
    im2col_gpu(input, batch, in_channels, height, width,
               kernel_h, kernel_w, pad_h, pad_w,
               stride_h, stride_w, height_col, width_col,
               col_buffer);

    float* grad_output_matrix;
    cudaMalloc(&grad_output_matrix, out_channels * n_cols * sizeof(float));
    nchw_to_matrix(grad_output, grad_output_matrix, batch, out_channels, height_col, width_col);

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta_zero = 0.0f;

    if (grad_weights != nullptr) {
        cublasSgemm(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    kernel_dim, out_channels, n_cols,
                    &alpha,
                    col_buffer, n_cols,
                    grad_output_matrix, n_cols,
                    &beta_zero,
                    grad_weights, kernel_dim);
    }

    if (grad_bias != nullptr) {
        float* ones;
        cudaMalloc(&ones, n_cols * sizeof(float));
        fill_device(ones, static_cast<size_t>(n_cols), 1.0f);
        cublasSgemv(handle,
                    CUBLAS_OP_T,
                    n_cols, out_channels,
                    &alpha,
                    grad_output_matrix, n_cols,
                    ones, 1,
                    &beta_zero,
                    grad_bias, 1);
        cudaFree(ones);
    }

    float* grad_col;
    cudaMalloc(&grad_col, kernel_dim * n_cols * sizeof(float));
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                n_cols, kernel_dim, out_channels,
                &alpha,
                grad_output_matrix, n_cols,
                weights, kernel_dim,
                &beta_zero,
                grad_col, n_cols);

    col2im_gpu(grad_col, batch, in_channels, height, width,
               kernel_h, kernel_w, pad_h, pad_w,
               stride_h, stride_w, height_col, width_col,
               grad_input);

    cublasDestroy(handle);
    cudaFree(grad_col);
    cudaFree(grad_output_matrix);
    cudaFree(col_buffer);
}

void test_conv_im2col() {
    int batch = 1;
    int in_channels = 1;
    int out_channels = 1;
    int height = 4;
    int width = 4;
    int kernel = 3;
    int pad = 1;

    int input_size = batch * in_channels * height * width;
    int weight_size = out_channels * in_channels * kernel * kernel;
    int output_size = batch * out_channels * height * width;

    std::vector<float> h_input(input_size);
    for (int i = 0; i < input_size; ++i) {
        h_input[i] = static_cast<float>(i + 1);
    }

    std::vector<float> h_weights = {
        1.f, 0.f, -1.f,
        1.f, 0.f, -1.f,
        1.f, 0.f, -1.f
    };
    std::vector<float> h_bias = {0.5f};

    std::vector<float> h_grad_output(output_size);
    for (int i = 0; i < output_size; ++i) {
        h_grad_output[i] = 0.1f * (i + 1);
    }

    float *d_input, *d_weights, *d_bias;
    float *d_output, *d_grad_output, *d_grad_input, *d_grad_weights, *d_grad_bias;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_weights, weight_size * sizeof(float));
    cudaMalloc(&d_bias, out_channels * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_grad_output, output_size * sizeof(float));
    cudaMalloc(&d_grad_input, input_size * sizeof(float));
    cudaMalloc(&d_grad_weights, weight_size * sizeof(float));
    cudaMalloc(&d_grad_bias, out_channels * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), out_channels * sizeof(float), cudaMemcpyHostToDevice);

    conv2d_forward(d_input, d_weights, d_bias, d_output,
                   batch, in_channels, out_channels, height, width, kernel, kernel, pad, pad, 1, 1);

    std::vector<float> h_output_gpu(output_size, 0.f);
    cudaMemcpy(h_output_gpu.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(d_grad_output, h_grad_output.data(), output_size * sizeof(float), cudaMemcpyHostToDevice);
    conv2d_backward(d_input, d_grad_output, d_weights,
                    d_grad_input, d_grad_weights, d_grad_bias,
                    batch, in_channels, out_channels, height, width, kernel, kernel, pad, pad, 1, 1);

    std::vector<float> h_grad_input_gpu(input_size, 0.f);
    std::vector<float> h_grad_weights_gpu(weight_size, 0.f);
    std::vector<float> h_grad_bias_gpu(out_channels, 0.f);
    cudaMemcpy(h_grad_input_gpu.data(), d_grad_input, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_weights_gpu.data(), d_grad_weights, weight_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_bias_gpu.data(), d_grad_bias, out_channels * sizeof(float), cudaMemcpyDeviceToHost);

    auto almost_equal = [](float a, float b) {
        return std::fabs(a - b) < 1e-4f;
    };

    bool passed = true;
    const float expected_forward[] = {
        -7.5f, -3.5f, -3.5f, 10.5f,
        -17.5f, -5.5f, -5.5f, 21.5f,
        -29.5f, -5.5f, -5.5f, 33.5f,
        -23.5f, -3.5f, -3.5f, 26.5f
    };
    for (int i = 0; i < output_size; ++i) {
        if (!almost_equal(h_output_gpu[i], expected_forward[i])) {
            std::cout << "Conv forward mismatch at index " << i
                      << ": expected " << expected_forward[i]
                      << ", got " << h_output_gpu[i] << std::endl;
            passed = false;
            break;
        }
    }

    const float expected_grad_input[] = {
        0.8f, 0.4f, 0.4f, -1.0f,
        1.8000001f, 0.6f, 0.6f, -2.1f,
        3.0f, 0.60000014f, 0.6f, -3.3f,
        2.4f, 0.4000001f, 0.4000001f, -2.6f
    };
    for (int i = 0; i < input_size && passed; ++i) {
        if (!almost_equal(h_grad_input_gpu[i], expected_grad_input[i])) {
            std::cout << "Grad input mismatch at index " << i
                      << ": expected " << expected_grad_input[i]
                      << ", got " << h_grad_input_gpu[i] << std::endl;
            passed = false;
            break;
        }
    }

    const float expected_grad_weights[] = {
        69.6f, 96.2f, 73.2f,
        111.2f, 149.6f, 111.200005f,
        73.2f, 96.2f, 69.6f
    };
    for (int i = 0; i < weight_size && passed; ++i) {
        if (!almost_equal(h_grad_weights_gpu[i], expected_grad_weights[i])) {
            std::cout << "Grad weight mismatch at index " << i
                      << ": expected " << expected_grad_weights[i]
                      << ", got " << h_grad_weights_gpu[i] << std::endl;
            passed = false;
            break;
        }
    }

    const float expected_grad_bias[] = {13.6f};
    for (int i = 0; i < out_channels && passed; ++i) {
        if (!almost_equal(h_grad_bias_gpu[i], expected_grad_bias[i])) {
            std::cout << "Grad bias mismatch at index " << i
                      << ": expected " << expected_grad_bias[i]
                      << ", got " << h_grad_bias_gpu[i] << std::endl;
            passed = false;
            break;
        }
    }

    if (passed) {
        printf("Conv im2col test passed!\n");
    }

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudaFree(d_grad_weights);
    cudaFree(d_grad_bias);
}


void forward_fc(float* input, float* output, float* weights, float* bias, 
                         int batch_size, int in_features, int out_features) { 
    cublasHandle_t handle; 
    cublasCreate(&handle); 
    
    const float alpha = 1.0f, beta = 0.0f; 
    
    cublasSgemm(handle, 
                CUBLAS_OP_T, CUBLAS_OP_N,           
                out_features, batch_size, in_features, 
                &alpha, 
                weights, in_features,                
                input, in_features,                  
                &beta, 
                output, out_features);               
    
    if (bias != nullptr) { 
        float* ones;
        cudaMalloc(&ones, batch_size * sizeof(float));
        fill_device(ones, static_cast<size_t>(batch_size), 1.0f);
        const float alpha_bias = 1.0f, beta_output = 1.0f;
        cublasSgemm(handle, 
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    out_features, batch_size, 1,
                    &alpha_bias,
                    bias, out_features,
                    ones, 1,
                    &beta_output,
                    output, out_features);
        
        cudaFree(ones);
    } 
    cublasDestroy(handle); 
}

void backward_fc(float* input, float* grad_output, float* weights,
                 float* grad_input, float* grad_weights, float* grad_bias,
                 int batch_size, int in_features, int out_features) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f, beta_zero = 0.0f;

    if (grad_input != nullptr) {
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    in_features, batch_size, out_features,
                    &alpha,
                    weights, in_features,
                    grad_output, out_features,
                    &beta_zero,
                    grad_input, in_features);
    }

    if (grad_weights != nullptr) {
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    in_features, out_features, batch_size,
                    &alpha,
                    input, in_features,
                    grad_output, out_features,
                    &beta_zero,
                    grad_weights, in_features);
    }

    if (grad_bias != nullptr) {
        float* ones;
        cudaMalloc(&ones, batch_size * sizeof(float));
        fill_device(ones, static_cast<size_t>(batch_size), 1.0f);

        cublasSgemv(handle,
                    CUBLAS_OP_N,
                    out_features, batch_size,
                    &alpha,
                    grad_output, out_features,
                    ones, 1,
                    &beta_zero,
                    grad_bias, 1);

        cudaFree(ones);
    }

    cublasDestroy(handle);
}

void test_forward_fc(){ 
    int batch_size = 2, in_features = 3, out_features = 4; 
    float h_input[] = {1, 2, 3,  
                       4, 5, 6};   
    float h_weights[] = {1, 0, 1,
                         0, 1, 1,
                         1, 1, 0,  
                         0, 0, 1}; 
    float h_bias[] = {0.5, 0.5, 0.5, 0.5}; 
    float h_output[8] = {0}; 

    float *d_input, *d_weights, *d_bias, *d_output; 
    cudaMalloc(&d_input, batch_size * in_features * sizeof(float)); 
    cudaMalloc(&d_weights, out_features * in_features * sizeof(float)); 
    cudaMalloc(&d_bias, out_features * sizeof(float)); 
    cudaMalloc(&d_output, batch_size * out_features * sizeof(float)); 
    
    cudaMemcpy(d_input, h_input, batch_size * in_features * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_weights, h_weights, out_features * in_features * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_bias, h_bias, out_features * sizeof(float), cudaMemcpyHostToDevice); 

    forward_fc(d_input, d_output, d_weights, d_bias, 
                        batch_size, in_features, out_features); 
    
    cudaMemcpy(h_output, d_output, batch_size * out_features * sizeof(float), cudaMemcpyDeviceToHost); 
    
    float expected_output[] = {4.5, 5.5, 3.5, 3.5, 
                               10.5, 11.5, 9.5, 6.5};
    
    int passed = 1;
    for(int i = 0; i < batch_size * out_features; i++){ 
        if(std::fabs(h_output[i] - expected_output[i]) > 1e-5){ 
            std::cout << "Test failed at index " << i << ": expected " << expected_output[i] << ", got " << h_output[i] << std::endl; 
            passed = 0;
        }
    }

    if (passed) {
        printf("Test1 passed!\n");
    }

    cudaFree(d_input); 
    cudaFree(d_weights); 
    cudaFree(d_bias); 
    cudaFree(d_output); 
} 

void test_backward_fc(){
    int batch_size = 2, in_features = 3, out_features = 4;
    float h_input[] = {1, 2, 3,
                       4, 5, 6};
    float h_weights[] = {1, 0, 1,
                         0, 1, 1,
                         1, 1, 0,
                         0, 0, 1};
    float h_grad_output[] = {0.1f, 0.2f, 0.3f, 0.4f,
                             0.5f, 0.6f, 0.7f, 0.8f};
    float h_grad_input[6] = {0};
    float h_grad_weights[12] = {0};
    float h_grad_bias[4] = {0};

    float *d_input, *d_weights, *d_grad_output;
    float *d_grad_input, *d_grad_weights, *d_grad_bias;
    cudaMalloc(&d_input, batch_size * in_features * sizeof(float));
    cudaMalloc(&d_weights, out_features * in_features * sizeof(float));
    cudaMalloc(&d_grad_output, batch_size * out_features * sizeof(float));
    cudaMalloc(&d_grad_input, batch_size * in_features * sizeof(float));
    cudaMalloc(&d_grad_weights, out_features * in_features * sizeof(float));
    cudaMalloc(&d_grad_bias, out_features * sizeof(float));

    cudaMemcpy(d_input, h_input, batch_size * in_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, out_features * in_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, h_grad_output, batch_size * out_features * sizeof(float), cudaMemcpyHostToDevice);

    backward_fc(d_input, d_grad_output, d_weights,
                d_grad_input, d_grad_weights, d_grad_bias,
                batch_size, in_features, out_features);

    cudaMemcpy(h_grad_input, d_grad_input, batch_size * in_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_weights, d_grad_weights, out_features * in_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_bias, d_grad_bias, out_features * sizeof(float), cudaMemcpyDeviceToHost);

    float expected_grad_input[] = {0.4f, 0.5f, 0.70000005f,
                                   1.2f, 1.3f, 1.9000001f};
    float expected_grad_weights[] = {2.1f, 2.7f, 3.3f,
                                     2.6f, 3.4f, 4.2f,
                                     3.1f, 4.1f, 5.1f,
                                     3.6f, 4.8f, 6.0f};
    float expected_grad_bias[] = {0.6f, 0.8f, 1.0f, 1.2f};

    int passed = 1;
    for (int i = 0; i < batch_size * in_features; ++i) {
        if (std::fabs(h_grad_input[i] - expected_grad_input[i]) > 1e-5f) {
            std::cout << "Grad input mismatch at index " << i
                      << ": expected " << expected_grad_input[i]
                      << ", got " << h_grad_input[i] << std::endl;
            passed = 0;
        }
    }

    for (int i = 0; i < out_features * in_features; ++i) {
        if (std::fabs(h_grad_weights[i] - expected_grad_weights[i]) > 1e-5f) {
            std::cout << "Grad weights mismatch at index " << i
                      << ": expected " << expected_grad_weights[i]
                      << ", got " << h_grad_weights[i] << std::endl;
            passed = 0;
        }
    }

    for (int i = 0; i < out_features; ++i) {
        if (std::fabs(h_grad_bias[i] - expected_grad_bias[i]) > 1e-5f) {
            std::cout << "Grad bias mismatch at index " << i
                      << ": expected " << expected_grad_bias[i]
                      << ", got " << h_grad_bias[i] << std::endl;
            passed = 0;
        }
    }

    if (passed) {
        printf("Test2 passed!\n");
    }

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudaFree(d_grad_weights);
    cudaFree(d_grad_bias);
}

__global__ void max_pool_forward_kernel(const float* in_data, float* out_data, float* out_mask,
                                        int nthreads, int num, int channels,
                                        int in_h, int in_w, int out_h, int out_w,
                                        int kernel_h, int kernel_w, int stride_h, int stride_w) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int n = index / (channels * out_h * out_w);
        int c = (index / (out_h * out_w)) % channels;
        int ph = (index / out_w) % out_h;
        int pw = index % out_w;

        int hstart = ph * stride_h;
        int wstart = pw * stride_w;

        float max_val = -FLT_MAX;
        int max_idx = -1;

        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h = hstart + kh;
                int w = wstart + kw;
                int in_index = ((n * channels + c) * in_h + h) * in_w + w;
                float val = in_data[in_index];
                if (val > max_val) {
                    max_val = val;
                    max_idx = in_index;
                }
            }
        }

        out_data[index] = max_val;
        if (out_mask != nullptr) {
            out_mask[index] = static_cast<float>(max_idx);
        }
    }
}

__global__ void max_pool_backward_kernel(const float* grad_out, const float* mask,
                                         float* grad_in, int nthreads) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int in_index = static_cast<int>(mask[index]);
        if (in_index >= 0) {
            atomicAdd(&grad_in[in_index], grad_out[index]);
        }
    }
}

void max_pool_forward_layer(float* input, float* output, float* mask,
                            int num, int channels, int height, int width) {
    const int kernel_h = 2, kernel_w = 2;
    const int stride_h = 2, stride_w = 2;
    int out_h = height / 2;
    int out_w = width / 2;
    int nthreads = num * channels * out_h * out_w;
    int threads = 256;
    int blocks = (nthreads + threads - 1) / threads;
    max_pool_forward_kernel<<<blocks, threads>>>(input, output, mask,
                                                 nthreads, num, channels,
                                                 height, width, out_h, out_w,
                                                 kernel_h, kernel_w, stride_h, stride_w);
}

void max_pool_backward_layer(float* grad_out, float* mask, float* grad_in,
                             int num, int channels, int height, int width) {
    int out_h = height / 2;
    int out_w = width / 2;
    int nthreads = num * channels * out_h * out_w;
    cudaMemset(grad_in, 0, num * channels * height * width * sizeof(float));
    int threads = 256;
    int blocks = (nthreads + threads - 1) / threads;
    max_pool_backward_kernel<<<blocks, threads>>>(grad_out, mask, grad_in, nthreads);
}

void test_max_pool_forward() {
    int num = 1, channels = 1, height = 4, width = 4;
    float h_input[] = {
        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
        9.f, 10.f, 11.f, 12.f,
        13.f, 14.f, 15.f, 16.f
    };
    float expected_output[] = {6.f, 8.f, 14.f, 16.f};

    float *d_input, *d_output, *d_mask;
    cudaMalloc(&d_input, num * channels * height * width * sizeof(float));
    cudaMalloc(&d_output, num * channels * (height / 2) * (width / 2) * sizeof(float));
    cudaMalloc(&d_mask, num * channels * (height / 2) * (width / 2) * sizeof(float));

    cudaMemcpy(d_input, h_input, num * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    max_pool_forward_layer(d_input, d_output, d_mask, num, channels, height, width);

    float h_output[4] = {0};
    cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost);

    int passed = 1;
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(h_output[i] - expected_output[i]) > 1e-5f) {
            std::cout << "MaxPool forward mismatch at index " << i
                      << ": expected " << expected_output[i]
                      << ", got " << h_output[i] << std::endl;
            passed = 0;
        }
    }

    if (passed) {
        printf("MaxPool forward test passed!\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
}

void test_max_pool_backward() {
    int num = 1, channels = 1, height = 4, width = 4;
    float h_input[] = {
        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
        9.f, 10.f, 11.f, 12.f,
        13.f, 14.f, 15.f, 16.f
    };
    float h_grad_out[] = {1.f, 2.f, 3.f, 4.f};
    float expected_grad_in[] = {
        0.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 2.f,
        0.f, 0.f, 0.f, 0.f,
        0.f, 3.f, 0.f, 4.f
    };

    size_t input_bytes = num * channels * height * width * sizeof(float);
    size_t output_bytes = num * channels * (height / 2) * (width / 2) * sizeof(float);

    float *d_input, *d_output, *d_mask;
    float *d_grad_out, *d_grad_in;
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_output, output_bytes);
    cudaMalloc(&d_mask, output_bytes);
    cudaMalloc(&d_grad_out, output_bytes);
    cudaMalloc(&d_grad_in, input_bytes);

    cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);
    max_pool_forward_layer(d_input, d_output, d_mask, num, channels, height, width);

    cudaMemcpy(d_grad_out, h_grad_out, output_bytes, cudaMemcpyHostToDevice);
    max_pool_backward_layer(d_grad_out, d_mask, d_grad_in, num, channels, height, width);

    float h_grad_in[16] = {0};
    cudaMemcpy(h_grad_in, d_grad_in, input_bytes, cudaMemcpyDeviceToHost);

    int passed = 1;
    for (int i = 0; i < 16; ++i) {
        if (std::fabs(h_grad_in[i] - expected_grad_in[i]) > 1e-5f) {
            std::cout << "MaxPool backward mismatch at index " << i
                      << ": expected " << expected_grad_in[i]
                      << ", got " << h_grad_in[i] << std::endl;
            passed = 0;
        }
    }

    if (passed) {
        printf("MaxPool backward test passed!\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
    cudaFree(d_grad_out);
    cudaFree(d_grad_in);
}

__global__ void row_max_kernel(const float* input, float* row_max, int rows, int cols) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    float max_val = -FLT_MAX;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        int idx = row * cols + col;
        max_val = fmaxf(max_val, input[idx]);
    }

    shared[threadIdx.x] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        row_max[row] = shared[0];
    }
}

__global__ void subtract_max_kernel(const float* input, const float* row_max,
                                    float* output, int rows, int cols) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (index >= total) {
        return;
    }
    int row = index / cols;
    output[index] = input[index] - row_max[row];
}

__global__ void exp_kernel(float* data, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    data[index] = expf(data[index]);
}

__global__ void row_sum_kernel(const float* input, float* row_sum, int rows, int cols) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    float sum_val = 0.f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        int idx = row * cols + col;
        sum_val += input[idx];
    }

    shared[threadIdx.x] = sum_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        row_sum[row] = shared[0];
    }
}

__global__ void normalize_kernel(float* data, const float* row_sum, int rows, int cols) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (index >= total) {
        return;
    }
    int row = index / cols;
    float denom = row_sum[row];
    data[index] = (denom > 0.f) ? (data[index] / denom) : 0.f;
}

void softmax_forward(const float* input, float* output, int batch_size, int num_classes) {
    int total = batch_size * num_classes;
    float* d_row_max;
    float* d_row_sum;
    cudaMalloc(&d_row_max, batch_size * sizeof(float));
    cudaMalloc(&d_row_sum, batch_size * sizeof(float));

    int threads_per_block = 256;
    size_t shared_mem = threads_per_block * sizeof(float);

    row_max_kernel<<<batch_size, threads_per_block, shared_mem>>>(input, d_row_max, batch_size, num_classes);

    int blocks = (total + threads_per_block - 1) / threads_per_block;
    subtract_max_kernel<<<blocks, threads_per_block>>>(input, d_row_max, output, batch_size, num_classes);
    exp_kernel<<<blocks, threads_per_block>>>(output, total);

    row_sum_kernel<<<batch_size, threads_per_block, shared_mem>>>(output, d_row_sum, batch_size, num_classes);

    normalize_kernel<<<blocks, threads_per_block>>>(output, d_row_sum, batch_size, num_classes);

    cudaFree(d_row_max);
    cudaFree(d_row_sum);
}

void test_softmax_forward() {
    int batch = 2;
    int classes = 3;
    float h_input[] = {1.f, 2.f, 3.f,
                       1.f, 2.f, 4.f};

    float expected_output[] = {
        0.09003058f, 0.24472848f, 0.66524094f,
        0.04201007f, 0.1141952f, 0.8437947f
    };

    float *d_input, *d_output;
    cudaMalloc(&d_input, batch * classes * sizeof(float));
    cudaMalloc(&d_output, batch * classes * sizeof(float));

    cudaMemcpy(d_input, h_input, batch * classes * sizeof(float), cudaMemcpyHostToDevice);

    softmax_forward(d_input, d_output, batch, classes);

    float h_output[6] = {0};
    cudaMemcpy(h_output, d_output, batch * classes * sizeof(float), cudaMemcpyDeviceToHost);

    int passed = 1;
    for (int i = 0; i < batch * classes; ++i) {
        if (std::fabs(h_output[i] - expected_output[i]) > 1e-5f) {
            std::cout << "Softmax mismatch at index " << i
                      << ": expected " << expected_output[i]
                      << ", got " << h_output[i] << std::endl;
            passed = 0;
        }
    }

    if (passed) {
        printf("Softmax forward test passed!\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

__global__ void cross_entropy_loss_kernel(const float* probs, const int* labels,
                                          float* loss_buffer, int batch_size, int num_classes) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size) {
        return;
    }
    int label = labels[index];
    float prob = probs[index * num_classes + label];
    prob = fmaxf(prob, 1e-12f);
    loss_buffer[index] = -logf(prob);
}

float cross_entropy_loss_forward(const float* probs, const int* labels,
                                 int batch_size, int num_classes) {
    float* d_loss_buffer;
    cudaMalloc(&d_loss_buffer, batch_size * sizeof(float));

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    cross_entropy_loss_kernel<<<blocks, threads>>>(probs, labels, d_loss_buffer, batch_size, num_classes);

    std::vector<float> h_losses(batch_size, 0.f);
    cudaMemcpy(h_losses.data(), d_loss_buffer, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    float total_loss = 0.f;
    for (float val : h_losses) {
        total_loss += val;
    }

    cudaFree(d_loss_buffer);
    return total_loss / batch_size;
}

__global__ void softmax_cross_entropy_backward_kernel(const float* probs, const int* labels,
                                                      float* grad_input, int batch_size, int num_classes) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_classes;
    if (index >= total) {
        return;
    }
    int row = index / num_classes;
    int col = index % num_classes;
    int label = labels[row];
    float grad = probs[index];
    if (col == label) {
        grad -= 1.f;
    }
    grad_input[index] = grad / batch_size;
}

void cross_entropy_with_softmax_backward(const float* probs, const int* labels,
                                         float* grad_input, int batch_size, int num_classes) {
    int total = batch_size * num_classes;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    softmax_cross_entropy_backward_kernel<<<blocks, threads>>>(probs, labels, grad_input, batch_size, num_classes);
}

void test_softmax_cross_entropy() {
    int batch = 2;
    int classes = 3;
    float h_input[] = {1.f, 2.f, 3.f,
                       1.f, 2.f, 4.f};
    int h_labels[] = {2, 1};

    float expected_loss = 1.288726f;
    float expected_grad[] = {
        0.04501529f, 0.12236424f, -0.16737953f,
        0.02100503f, -0.4429024f, 0.42189735f
    };

    float *d_input, *d_probs, *d_grad;
    int *d_labels;
    cudaMalloc(&d_input, batch * classes * sizeof(float));
    cudaMalloc(&d_probs, batch * classes * sizeof(float));
    cudaMalloc(&d_grad, batch * classes * sizeof(float));
    cudaMalloc(&d_labels, batch * sizeof(int));

    cudaMemcpy(d_input, h_input, batch * classes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, batch * sizeof(int), cudaMemcpyHostToDevice);

    softmax_forward(d_input, d_probs, batch, classes);
    float loss = cross_entropy_loss_forward(d_probs, d_labels, batch, classes);

    cross_entropy_with_softmax_backward(d_probs, d_labels, d_grad, batch, classes);

    float h_grad[6] = {0};
    cudaMemcpy(h_grad, d_grad, batch * classes * sizeof(float), cudaMemcpyDeviceToHost);

    int passed = 1;
    if (std::fabs(loss - expected_loss) > 1e-5f) {
        std::cout << "Cross entropy loss mismatch: expected " << expected_loss
                  << ", got " << loss << std::endl;
        passed = 0;
    }

    for (int i = 0; i < batch * classes; ++i) {
        if (std::fabs(h_grad[i] - expected_grad[i]) > 1e-5f) {
            std::cout << "Cross entropy grad mismatch at index " << i
                      << ": expected " << expected_grad[i]
                      << ", got " << h_grad[i] << std::endl;
            passed = 0;
        }
    }

    if (passed) {
        printf("Softmax + CrossEntropy test passed!\n");
    }

    cudaFree(d_input);
    cudaFree(d_probs);
    cudaFree(d_grad);
    cudaFree(d_labels);
}

int main(){ 
    test_forward_fc();
    test_backward_fc();
    test_max_pool_forward();
    test_max_pool_backward();
    test_softmax_cross_entropy();
    test_conv_im2col();
    test_softmax_forward();
    test_softmax_cross_entropy();
    return 0; 
} 
