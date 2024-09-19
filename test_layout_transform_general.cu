#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_RANK 8

// Scatter version: Threads map over the input array
__global__ void layout_transform_scatter(float* input, float* output, int total_elements, int rank, int32_t* input_strides, int32_t* output_strides, int32_t* axes_order) {
    int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (input_idx >= total_elements) return;

    int input_coords[MAX_RANK];
    int output_coords[MAX_RANK];

    // Compute input_coords from input_idx
    int idx = input_idx;
    for (int i = 0; i < rank; i++) {
        input_coords[i] = idx / input_strides[i];
        idx = idx % input_strides[i];
    }

    // Map input_coords to output_coords via axes_order
    for (int i = 0; i < rank; i++) {
        output_coords[i] = input_coords[axes_order[i]];
    }

    // Compute output_idx
    int output_idx = 0;
    for (int i = 0; i < rank; i++) {
        output_idx += output_coords[i] * output_strides[i];
    }

    // Copy data
    output[output_idx] = input[input_idx];
}

// Gather version: Threads map over the output array
__global__ void layout_transform_gather(float* input, float* output, int total_elements, int rank, int32_t* input_strides, int32_t* output_strides, int32_t* axes_order_inv) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= total_elements) return;

    int output_coords[MAX_RANK];
    int input_coords[MAX_RANK];

    // Compute output_coords from output_idx
    int idx = output_idx;
    for (int i = 0; i < rank; i++) {
        output_coords[i] = idx / output_strides[i];
        idx = idx % output_strides[i];
    }

    // Map output_coords to input_coords via axes_order_inv
    for (int i = 0; i < rank; i++) {
        input_coords[i] = output_coords[axes_order_inv[i]];
    }

    // Compute input_idx
    int input_idx = 0;
    for (int i = 0; i < rank; i++) {
        input_idx += input_coords[i] * input_strides[i];
    }

    // Copy data
    output[output_idx] = input[input_idx];
}

void launch_transform(float* d_in, float* d_out, int32_t* h_input_shape, int32_t* h_axes_order, int rank, bool scatter) {
    int total_elements = 1;
    for (int i = 0; i < rank; i++) {
        total_elements *= h_input_shape[i];
    }

    int32_t h_input_strides[MAX_RANK];
    int32_t h_output_shape[MAX_RANK];
    int32_t h_output_strides[MAX_RANK];
    int32_t h_axes_order_inv[MAX_RANK];

    // Compute input_strides
    h_input_strides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
        h_input_strides[i] = h_input_strides[i + 1] * h_input_shape[i + 1];
    }

    // Compute output_shape
    for (int i = 0; i < rank; i++) {
        h_output_shape[i] = h_input_shape[h_axes_order[i]];
    }

    // Compute output_strides
    h_output_strides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
        h_output_strides[i] = h_output_strides[i + 1] * h_output_shape[i + 1];
    }

    // Compute axes_order_inv
    for (int i = 0; i < rank; i++) {
        h_axes_order_inv[h_axes_order[i]] = i;
    }

    // Allocate device memory for strides and axes_order
    int32_t* d_input_strides;
    int32_t* d_output_strides;
    int32_t* d_axes_order;
    int32_t* d_axes_order_inv;

    cudaMalloc(&d_input_strides, rank * sizeof(int32_t));
    cudaMalloc(&d_output_strides, rank * sizeof(int32_t));
    cudaMalloc(&d_axes_order, rank * sizeof(int32_t));
    cudaMalloc(&d_axes_order_inv, rank * sizeof(int32_t));

    cudaMemcpy(d_input_strides, h_input_strides, rank * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_strides, h_output_strides, rank * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_axes_order, h_axes_order, rank * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_axes_order_inv, h_axes_order_inv, rank * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    if (scatter) {
        layout_transform_scatter<<<num_blocks, threads_per_block>>>(d_in, d_out, total_elements, rank, d_input_strides, d_output_strides, d_axes_order);
    } else {
        layout_transform_gather<<<num_blocks, threads_per_block>>>(d_in, d_out, total_elements, rank, d_input_strides, d_output_strides, d_axes_order_inv);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Clean up
    cudaFree(d_input_strides);
    cudaFree(d_output_strides);
    cudaFree(d_axes_order);
    cudaFree(d_axes_order_inv);
}

int main() {
    int m = 16;
    int k = 16; 
    int size = m * k;

    // Define rank and shapes
    int rank = 4;
    int32_t h_input_shape[MAX_RANK] = { m / 8, 8, k / 8, 8 };
    int32_t h_axes_order[MAX_RANK] = { 0, 2, 1, 3 };

    float* h_in = (float*)malloc(size * sizeof(float));
    float* h_out_scatter = (float*)malloc(size * sizeof(float));
    float* h_out_gather = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) {
        h_in[i] = i;
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));

    cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch scatter transform
    launch_transform(d_in, d_out, h_input_shape, h_axes_order, rank, true);

    cudaMemcpy(h_out_scatter, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Launch gather transform
    launch_transform(d_in, d_out, h_input_shape, h_axes_order, rank, false);

    cudaMemcpy(h_out_gather, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    printf("Scatter Transform Output:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            printf("%.0f ", h_out_scatter[i * k + j]);
        }
        printf("\n");
    }

    printf("\nGather Transform Output:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            printf("%.0f ", h_out_gather[i * k + j]);
        }
        printf("\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out_scatter);
    free(h_out_gather);

    return 0;
}
