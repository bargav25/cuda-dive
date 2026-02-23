#include <stdio.h>

__global__ void hello_kernel(int *out) {

    // thread 0 writes a value

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 25;
    }
}

int main() {

    // d_out is pointer to an empty array
    int *d_out = nullptr;

    // Stack variable on Host side 
    int h_out = 0;

    // Allocate one int to d_out on GPU
    cudaMalloc(&d_out, sizeof(int));

    // Launch the kernel with <<< grid_size, block_size >>>
    hello_kernel<<<1, 1>>>(d_out);

    // Wait for kernel to finish, can also catch runtime errors
    cudaDeviceSynchronize();

    // Copy back to CPU (Copy 4 bytes into the variable h_out’s memory address)
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Kernel wrote : %d\n", h_out);

    cudaFree(d_out);

    return 0;

}