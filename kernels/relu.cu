#include <stdio.h>


__global__ void relu(float *arr, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Divergence inside warp
    if (i < n) {
        if (arr[i] < 0) arr[i] = 0;
    }
}

float* createArray(int n) {

    // allocating mem in Host
    float *arr = (float*)malloc(n * sizeof(float));

    // Tryna create some random float array with neg nums too
    for (int i = 0; i < n; i++) {
        if (i%2 == 0) arr[i] = i + i * 1.5;
        else arr[i] = i - i * 1.5;
    }

    return arr; // Returns pointer to heap memory

}


void print_arr(float *arr, int n) {

    for (int i=0; i < n; i++) printf("%.1f ", arr[i]);
    printf("\n");
}



int main() {

    int n = 30;

    // Save required bytes once and for all 
    // size_t is an unsigned integer data type in C/C++ used to represent the size of objects in bytes
    size_t req_bytes = n * sizeof(float);

    float *h_arr = createArray(n); // Host side array

    // Print the array
    printf("Input: \n");
    print_arr(h_arr, n);


    float *d_arr; // Device side array

    cudaMalloc(&d_arr, req_bytes); // Allocating mem in Device

    // Copy values from host to device
    cudaMemcpy(d_arr, h_arr, req_bytes, cudaMemcpyHostToDevice);

    int num_threads = 16;
    int num_blocks = (n + num_threads - 1) / num_threads;

    // Launch kernel
    relu<<<num_blocks, num_threads>>>(d_arr, n);

    // Synchronize
    cudaDeviceSynchronize();

    // Copy values from device to host
    cudaMemcpy(h_arr, d_arr, req_bytes, cudaMemcpyDeviceToHost);

    // Print the array
    printf("Output: \n");
    print_arr(h_arr, n);

    // Free up the space
    cudaFree(d_arr);
    free(h_arr);

    return 0;



}