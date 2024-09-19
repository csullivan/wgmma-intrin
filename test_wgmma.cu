#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>

#include "layout_transform.cuh"

// Forward declarations
void wgmma_kernel(void* a, void* b, void* c, int m, int n, int k);
__global__ void mma_naive(__nv_bfloat16* a, __nv_bfloat16* b, __nv_bfloat16* c);

// Helper function to convert float to __nv_bfloat16
__nv_bfloat16 float_to_bf16(float f) {
    return __float2bfloat16(f);
}

// Add this function to check if two values are close
bool is_close(float a, float b, float rtol = 1e-5, float atol = 1e-8) {
    return std::fabs(a - b) <= (atol + rtol * std::fabs(b));
}

int main() {
    const int m = 64, n = 256, k = 16;
    const int size_a = m * k;
    const int size_b = k * n;
    const int size_c = m * n;

    // Allocate host memory
    std::vector<__nv_bfloat16> h_a(size_a);
    std::vector<__nv_bfloat16> h_b(size_b);
    std::vector<__nv_bfloat16> h_c(size_c);

    // Initialize h_a and h_b with some values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < size_a; ++i) {
        h_a[i] = float_to_bf16(dis(gen));
        // h_a[i] = float_to_bf16(1.0f);
    }
    for (int i = 0; i < size_b; ++i) {
        h_b[i] = float_to_bf16(dis(gen));
        // h_b[i] = float_to_bf16(1.0f);
    }

    // Allocate device memory
    __nv_bfloat16 *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a * sizeof(__nv_bfloat16));
    cudaMalloc(&d_b, size_b * sizeof(__nv_bfloat16));
    cudaMalloc(&d_c, size_c * sizeof(__nv_bfloat16));

    // Copy data to device
    cudaMemcpy(d_a, h_a.data(), size_a * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size_b * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    // Allocate memory for transformed matrices
    __nv_bfloat16 *d_a_transformed, *d_b_transformed;
    cudaMalloc(&d_a_transformed, size_a * sizeof(__nv_bfloat16));
    cudaMalloc(&d_b_transformed, size_b * sizeof(__nv_bfloat16));

    // Define shapes and axes orders for transformations
    int32_t h_a_shape[4] = {m / 8, 8, k / 8, 8};
    int32_t h_a_axes_order[4] = {0, 2, 1, 3};
    int32_t h_b_shape[4] = {k / 8, 8, n / 8, 8};
    int32_t h_b_axes_order[4] = {2, 0, 3, 1};

    // Perform layout transformations
    launch_transform((float*)d_a, (float*)d_a_transformed, h_a_shape, h_a_axes_order, 4, true);
    launch_transform((float*)d_b, (float*)d_b_transformed, h_b_shape, h_b_axes_order, 4, true);

    // Allocate memory for naive result
    std::vector<__nv_bfloat16> h_c_naive(size_c);
    __nv_bfloat16 *d_c_naive;
    cudaMalloc(&d_c_naive, size_c * sizeof(__nv_bfloat16));

    // Perform WGMMA
    wgmma_kernel(d_a_transformed, d_b_transformed, d_c, m, n, k);

    // Perform naive matrix multiplication
    mma_naive<<<1, 128>>>(d_a, d_b, d_c_naive);

    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_c.data(), d_c, size_c * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_naive.data(), d_c_naive, size_c * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Compare results
    int num_mismatches = 0;
    for (int i = 0; i < size_c; ++i) {
        float wgmma_val = __bfloat162float(h_c[i]);
        float naive_val = __bfloat162float(h_c_naive[i]);
        if (!is_close(wgmma_val, naive_val)) {
            num_mismatches++;
            if (num_mismatches <= 10) {
                std::cout << "Mismatch at index " << i << ": WGMMA = " << wgmma_val 
                          << ", Naive = " << naive_val << std::endl;
            }
        }
    }

    // Print comparison results
    if (num_mismatches == 0) {
        std::cout << "All values match within tolerance!" << std::endl;
    } else {
        std::cout << "Total mismatches: " << num_mismatches << " out of " << size_c << " elements" << std::endl;
    }

    // Print a small portion of both results (e.g., top-left 4x4 corner)
    std::cout << "Top-left 4x4 corner of the WGMMA result:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = n-4; j < n; ++j) {
            std::cout << __bfloat162float(h_c[i * n + j]) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Top-left 4x4 corner of the naive result:" << std::endl;
    for (int i = m-4; i < m; ++i) {
        for (int j = n-4; j < n; ++j) {
            std::cout << __bfloat162float(h_c_naive[i * n + j]) << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_c_naive);
    cudaFree(d_a_transformed);
    cudaFree(d_b_transformed);

    return 0;
}


