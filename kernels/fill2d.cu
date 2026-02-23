
#include <stdio.h>

__global__ void fill_2d(int *arr, int width, int height) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        arr[row * width + col] = row * 1000 + col; // easy to visually verify
    }

}

int main() {

    int width = 13;
    int height = 7;

    size_t req_bytes = width * height * sizeof(int);

    // Initializing arrays and allocating space in host and device
    int *h = (int*)malloc(req_bytes);
    int *d = nullptr;
    cudaMalloc(&d, req_bytes);

    // Number of blocks, Number of Thread per block
    dim3 numThreads(16, 16);
    // In CUDA, blockIdx.x maps to columns, blockIdx,y maps to rows
    dim3 numBlocks((width + numThreads.x - 1) / numThreads.x , (height + numThreads.y - 1) / numThreads.y);

    // Launching the kernel
    fill_2d<<<numBlocks, numThreads>>>(d, width, height);

    // Synchronize
    cudaDeviceSynchronize();

    // Copying the values from device to host
    cudaMemcpy(h, d, req_bytes, cudaMemcpyDeviceToHost);

    // Print the matrix
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            printf("%05d ", h[r * width + c]);
        }
        printf("\n");
    }

    // Free up the space
    cudaFree(d);
    free(h);

    return 0;
}