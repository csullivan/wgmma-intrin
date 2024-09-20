#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>

#include "layout_transform.cuh"
#include "wgmma.cuh"

__nv_bfloat16 float_to_bf16(float f) {
    return __float2bfloat16(f);
}

bool is_close(float a, float b, float rtol = 1e-5, float atol = 1e-8) {
    return std::fabs(a - b) <= (atol + rtol * std::fabs(b));
}

int main() {
    const int m = 64, n = 256, k = 16;
    const int size_a = m * k;
    const int size_b = k * n;
    const int size_c = m * n;

    std::vector<__nv_bfloat16> h_a(size_a);
    std::vector<__nv_bfloat16> h_b(size_b);
    std::vector<__nv_bfloat16> h_c(size_c);

    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            if (i < 8 && j < 8) {
                float value = (i * 8 + j); 
                h_a[i * k + j] = float_to_bf16(value);
            } else {
                h_a[i * k + j] = float_to_bf16(0.0f);
            }
        }
    }
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i < 8 && j < 8) {
                float value = j * 8 + i;
                h_b[i * n + j] = float_to_bf16(value);
            } else {
                h_b[i * n + j] = float_to_bf16(0.0f);
            }
        }
    }

    __nv_bfloat16 *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a * sizeof(__nv_bfloat16));
    cudaMalloc(&d_b, size_b * sizeof(__nv_bfloat16));
    cudaMalloc(&d_c, size_c * sizeof(__nv_bfloat16));

    cudaMemcpy(d_a, h_a.data(), size_a * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size_b * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    __nv_bfloat16 *d_a_transformed, *d_b_transformed;
    cudaMalloc(&d_a_transformed, size_a * sizeof(__nv_bfloat16));
    cudaMalloc(&d_b_transformed, size_b * sizeof(__nv_bfloat16));

    // (m, k) -> (m//8, k//8, m%8, k%8)
    int32_t h_a_shape[4] = {m / 8, 8, k / 8, 8};
    int32_t h_a_axes_order[4] = {0, 2, 1, 3};

    // (k, n) -> (n//8, k//8, n%8, k%8) 
    int32_t h_b_shape[4] = {k / 8, 8, n / 8, 8};
    int32_t h_b_axes_order[4] = {2, 0, 3, 1};

    /* 
    (128 threads, 128 regs) -> 
    (
      4 tiles across all threads, 
      8 rows per tile, 
      4 threads per tile row, 
      32 cols across all regs, 
      2 reg groups per tile, 
      2 contiguous columns per register
    )
    */
    int32_t h_c_shape[6] = {4, 8, 4, 32, 2, 2};
    int32_t h_c_axes_order[6] = {0, 4, 1, 3, 2, 5};

    // Perform layout transforms for inputs
    launch_transform((__nv_bfloat16*)d_a, (__nv_bfloat16*)d_a_transformed, h_a_shape, h_a_axes_order, 4, true);
    launch_transform((__nv_bfloat16*)d_b, (__nv_bfloat16*)d_b_transformed, h_b_shape, h_b_axes_order, 4, true);

    std::vector<__nv_bfloat16> h_a_transformed(size_a);
    std::vector<__nv_bfloat16> h_b_transformed(size_b);

    cudaMemcpy(h_a_transformed.data(), d_a_transformed, size_a * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_transformed.data(), d_b_transformed, size_b * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    std::cout << "\nTop right corner of original A (16x16):" << std::endl;    
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            std::cout << __bfloat162float(h_a[i * k + j]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "\nTop right corner of transformed A (16x16):" << std::endl;    
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            std::cout << __bfloat162float(h_a_transformed[i * k + j]) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nTop right corner of original B (16x16):" << std::endl;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            std::cout << __bfloat162float(h_b[i * n + j]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "\nTop right corner of transformed B (1x66):" << std::endl;
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 66; ++j) {
            std::cout << __bfloat162float(h_b_transformed[i * n + j]) << " ";
        }
        std::cout << std::endl;
    }

    std::vector<__nv_bfloat16> h_c_naive(size_c);
    __nv_bfloat16 *d_c_naive;
    cudaMalloc(&d_c_naive, size_c * sizeof(__nv_bfloat16));

    wgmma_bf16_m64n256k16(d_a_transformed, d_b_transformed, d_c, m, n, k);

    __nv_bfloat16 *d_c_transformed;
    cudaMalloc(&d_c_transformed, size_c * sizeof(__nv_bfloat16));

    // Perform layout transform for the outputs
    launch_transform((__nv_bfloat16*)d_c, (__nv_bfloat16*)d_c_transformed, h_c_shape, h_c_axes_order, 6, false);

    cudaMemcpy(h_c.data(), d_c_transformed, size_c * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                sum += __bfloat162float(h_a[i * k + kk]) * __bfloat162float(h_b[kk * n + j]);
            }
            h_c_naive[i * n + j] = __float2bfloat16(sum);
        }
    }

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

    
    std::cout << "Top-left 4x4 corner of the WGMMA result:" << std::endl;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            std::cout << __bfloat162float(h_c[i * n + j]) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Top-left 4x4 corner of the naive result:" << std::endl;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            std::cout << __bfloat162float(h_c_naive[i * n + j]) << " ";
        }
        std::cout << std::endl;
    }
    if (num_mismatches == 0) {
        std::cout << "All values match within tolerance!" << std::endl;
    } else {
        std::cout << "Total mismatches: " << num_mismatches << " out of " << size_c << " elements" << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_c_naive);
    cudaFree(d_a_transformed);
    cudaFree(d_b_transformed);
    cudaFree(d_c_transformed);

    return 0;
}


