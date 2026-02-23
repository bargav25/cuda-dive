#include <stdio.h>

__global__ void assign_nums(int *arr, int n) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Since we launch threads more than we actually need
    if (i < n) arr[i] = i;

}

int main() {

    // Length of the array
    int n = 30;

    // Device side array, initialized with null pointer
    int *d_arr = nullptr;

    // Allocating memory in Host, to store the output
    int *h_arr = (int*)malloc(n * sizeof(int));

    // Allocating memory in Device, for the d_arr
    cudaMalloc(&d_arr, n * sizeof(int));

    // Choosing grid_size (or num_blocks) and block_size (or num_threads_per_block)
    int num_threads_per_block = 8;
    int num_blocks = (n + num_threads_per_block - 1) / num_threads_per_block;

    // Launching the kernel with <<< num_blocks, num_threads_per_block >>>
    assign_nums<<<num_blocks, num_threads_per_block>>>(d_arr, n);

    // Synchronize (wait for the kernel to finish)
    cudaDeviceSynchronize();

    // Copy the arr to Host
    cudaMemcpy(h_arr, d_arr, n*sizeof(int), cudaMemcpyDeviceToHost);

    // Printing the output
    for (int i = 0; i < n; i++) printf("%d ", h_arr[i]);

    printf("\n");

    // Freeing up allocations
    cudaFree(d_arr);
    free(h_arr);

    return 0;

}