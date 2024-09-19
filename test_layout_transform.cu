#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_DIM 8


// (m, k) -> (m // TILE_DIM, k // TILE_DIM, m % TILE_DIM, k % TILE_DIM)
__global__ void layout_transform_8x8(float* x, float* out, int m, int k) {
    int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = m * k;

    if (input_idx >= total_elements) return; // Boundary check

    int m_idx = input_idx / k;
    int k_idx = input_idx % k;

    // int num_tiles_m = m / TILE_DIM;
    int num_tiles_k = k / TILE_DIM;
    int m_tile_id = m_idx / TILE_DIM;
    int m_tile_idx = m_idx % TILE_DIM;
    int k_tile_id = k_idx / TILE_DIM;
    int k_tile_idx = k_idx % TILE_DIM;
    int output_idx = k_tile_idx + TILE_DIM*m_tile_idx + TILE_DIM*TILE_DIM*k_tile_id + num_tiles_k*TILE_DIM*TILE_DIM*m_tile_id;
    out[output_idx] = x[input_idx];
}



void launch_transform(float* d_in, float* d_out, int dim1, int dim2, bool core_a) {
    int total_elements = dim1 * dim2;
    int threads_per_block = 1024;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    if (core_a) {
        layout_transform_8x8<<<num_blocks, threads_per_block>>>(d_in, d_out, dim1, dim2);
    } else {
        // to_core_b<<<num_blocks, threads_per_block>>>(d_in, d_out, dim1, dim2);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

int main() {
    int m = 16;
    int k = 16; 
    int size = m * k;

    float* h_in = (float*)malloc(size * sizeof(float));
    float* h_out = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) {
        h_in[i] = i;
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));

    cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice);

    launch_transform(d_in, d_out, m, k, true);
    cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            printf("%f ", h_out[i * k + j]);
        }
        printf("\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
