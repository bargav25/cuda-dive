#include <stdio.h>

__global__ void whoami() {

    // If grid cube is flattened, gives us bx * by * bz blocks arranged in linear manner
    // Finding the position of the current block in that linear order
    // This is wrong: int block_id = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
    // Because CUDA grids are laid out in row-major order, where blockIdx.x changes fastest, then blockIdx.y and then blockIdx.z
    int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;

    // Same way to find thread id in the current block
    int thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;

    // absolute id = block_id * num_threads_per_block + thread_id
    int id = block_id * blockDim.x * blockDim.y * blockDim.z + thread_id;

    printf("%04d | Block (%d, %d, %d) = %03d | Thread (%d, %d, %d) = %03d\n", 
        id, blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_id);
    
}


int main() {

    // Grid sizes
    const int bx = 2, by = 3, bz = 4;

    // Block sizes
    const int tx = 2, ty = 4, tz = 8; // Since warp contain 32 threads, we will get two warps per block

    // Printing sizes
    printf("Number of blocks per grid: %d\n", bx * by * bz);
    printf("Number of threads per blocks: %d\n", tx * ty * tz);

    // Initializing the dimensions for kernel
    dim3 blocksPerGrid(bx, by, bz); // Imagine 3d cube of shape (bx, by, bz)
    dim3 threadsPerBlock(tx, ty, tz); // Imagine 3d cube of shape (tx, ty, tz) inside each cell of above cube

    // Launch the kernel
    whoami<<<blocksPerGrid, threadsPerBlock>>>();

    // Cuda Synchronize (wait for the kernel to finish)

    cudaDeviceSynchronize();

    return 0;


}