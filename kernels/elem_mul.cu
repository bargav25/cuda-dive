# include <stdio.h>

__global__ void vec_mul(float *a, float *b, float *c, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) c[i] = a[i] * b[i];
}

int main() {

    int n = 100;

    // Save required bytes once and for all 
    // size_t is an unsigned integer data type in C/C++ used to represent the size of objects in bytes
    size_t req_bytes = n * sizeof(float);

    // Host side arrays
    float h_a[n], h_b[n], h_c[n];

    for (int i=0; i<n; i++) {
        h_a[i] = i + 2.33; // some random float
        h_b[i] = i - 0.43; // some random float
    }

    // Device side arrays
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, req_bytes);
    cudaMalloc(&d_b, req_bytes);
    cudaMalloc(&d_c, req_bytes);

    // Copy input values from Host side array to Device side array
    cudaMemcpy(d_a, h_a, req_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, req_bytes, cudaMemcpyHostToDevice);

    // Decide on num_blocks, num_threads
    int num_threads = 32;
    int num_blocks = (n + num_threads - 1) / num_threads;

    // Launch Kernel
    vec_mul<<<num_blocks, num_threads>>>(d_a, d_b, d_c, n);

    // Synchronize (wait till the kernel gets finished)
    cudaDeviceSynchronize();

    // Copy output values from Device to Host
    cudaMemcpy(h_c, d_c, req_bytes, cudaMemcpyDeviceToHost);

    // Printing the output

    for (int i = 0; i < n; i++) printf("%.1f ", h_c[i]);

    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;

}