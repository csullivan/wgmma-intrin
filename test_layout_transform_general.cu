#include "layout_transform.cuh"

int main() {
    int m = 16;
    int k = 256; 
    int size = m * k;

    // Define rank and shapes
    int tile_size = 8;
    int rank = 4;
    int32_t h_input_shape[MAX_RANK] = { m / tile_size, tile_size, k / tile_size, tile_size };
    // int32_t h_axes_order[MAX_RANK] = { 0, 2, 1, 3 };
    int32_t h_axes_order[MAX_RANK] = { 2, 0, 3, 1 };

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
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            printf("%.0f ", h_out_scatter[i * k + j]);
        }
        printf("\n");
    }

    printf("\nGather Transform Output:\n");
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
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
