#include <stdio.h>
#include <stdlib.h>

#include <cuda/pipeline>
#include <cooperative_groups.h>
namespace cg=cooperative_groups;

template<int T>
__global__ void tiled_matmul(float *A, float *B, float *C, int M, int N, int K) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // shared mem in the form of pipeline with 2 stages

    // Shared Memory Buffers
    __shared__ float sA[2][T][T];
    __shared__ float sB[2][T][T];

    // Pipeline state, tracks completion of async copies 
    // cuda::pipeline_shared_state is the actual memory and resource pool used for synchronization.
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipeline_state;
    // cuda::thread_scope_block specify that state is shared among all the threads in a block
    // stage_count = 2 : How many asynchronous stages can be in flight simultaneously 

    // cuda::pipeline --> The interface that threads use to interact with shared data
    auto pipeline = cuda::make_pipeline(cg::this_thread_block(), &pipeline_state); 
    // cg::this_thread_block() --> Identify the thread group
        

    // Prologue: Fill stage 0 with tile 0

    // Producer allocating a stage for Async Copying 
    pipeline.producer_acquire();

    // Copy Instruction (async)
    bool a_in_bounds = (row < M && 0 * T + threadIdx.x < K);

    cuda::memcpy_async(
        &sA[0][threadIdx.y][threadIdx.x], // destination: shared mem
        &A[row * K + 0 * T + threadIdx.x], // source: global mem
        cuda::aligned_size_t<4>{a_in_bounds ? 4u : 0u}, 
        pipeline // tag this copy to current stage
    );

    bool b_in_bounds = (0 * T + threadIdx.x < K && col < N);

    cuda::memcpy_async(
        &sB[0][threadIdx.y][threadIdx.x],
        &B[(0 * T + threadIdx.y) * N + col],
        cuda::aligned_size_t<4>{b_in_bounds ? 4u : 0u}, 
        pipeline // tag this copy to current stage
    );

    // Seal the stage, i.e, no more copies allowed
    pipeline.producer_commit();

    float acc = 0.0f; // Same as naive gemm, each thread writing the output of a single cell

    // Iterating over K dimensions in tiles of size 
    int num_tiles = (K + T - 1) / T;

    for (int t=0; t < num_tiles; t++) { // for each tile

        // We have two buffers, to determine which one to use to compute or load, depends on whether t is odd or even cuz of pingpong
        // At t = 0 (which is even), we already have buffer 0 loading data from above, so load in buffer 1
        // Formulating it using modulo
        int load_buf = (t+1) % 2;
        int comp_buf = t % 2;

        if (t+1 < num_tiles) {

            // Load in load buffer
            pipeline.producer_acquire(); // a new stage has been allocated

            bool a_in_bounds = (row < M && (t+1) * T + threadIdx.x < K);

            cuda::memcpy_async(
                &sA[load_buf][threadIdx.y][threadIdx.x],
                &A[row * K + (t+1) * T + threadIdx.x],
                cuda::aligned_size_t<4>{a_in_bounds ? 4u : 0u},
                pipeline // tag this copy to the current stage
            );

            bool b_in_bounds = ((t+1) * T + threadIdx.x < K && col < N);

            cuda::memcpy_async(
                &sB[load_buf][threadIdx.y][threadIdx.x],
                &B[((t+1) * T + threadIdx.y) * N + col],
                cuda::aligned_size_t<4>{b_in_bounds ? 4u : 0u},
                pipeline
            );

            // commit the stage
            pipeline.producer_commit();
        }

        // Block the oldest stage (so as to load the current tile for computation)
        pipeline.consumer_wait();

        // Multiply the two tiles
        // each thread accumulates its dot product using sA and sB
        for (int k = 0; k < T; k++) acc += sA[comp_buf][threadIdx.y][k] * sB[comp_buf][k][threadIdx.x];

        // Release the stage for reuse
        pipeline.consumer_release();


    }

    if (row < M && col < N) C[row * N + col] = acc;


}

void initialize_arrs(float *A, float *B, int m, int n, int k) {

    // Filling inputs with determinsitic values
    for (int i = 0; i < m*k; i++) A[i] = 0.01 * float(i % 97);
    for (int i = 0; i < n*k; i++) B[i] = 0.02 * float(i % 89);

}

float* matmul_cpu(float *A, float *B, int M, int N, int K) {

    float *C = (float*)malloc(M*N*sizeof(float));

    // C[row, col] = sum_k A[row, k] * B[k, col]
    for (int r = 0; r<M; r++) {
        for (int c = 0; c<N; c++) {
            float acc = 0.0f;
            for (int k=0; k<K; k++) acc += A[r*K + k] * B[k * N + c];
            C[r * N + c] = acc;
        }
    }

    return C;

}

void check_correctness(float *arrA, float *arrB, int size) {

    float max_abs_error = 0.0f;

    for (int i = 0; i < size; i++) {
        float err = abs(arrA[i] - arrB[i]);

        if (err > max_abs_error) max_abs_error = err;
    }

    printf("OK. max absolute error: %.6f\n", max_abs_error);

}

int main() {

    // Dimensions of matrices
    int m = 1024, n = 1024, k = 1024;

    // Bytes required
    size_t bytesA = m * k * sizeof(float);
    size_t bytesB = n * k * sizeof(float);
    size_t bytesC = m * n * sizeof(float);

    // Create arrays in Host
    float *hA = (float*)malloc(bytesA);
    float *hB = (float*)malloc(bytesB);
    float *hC = (float*)malloc(bytesC);

    // Initialize some values in Input matrices
    initialize_arrs(hA, hB, m, n, k);

    // Device buffers
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;

    // Allocating memory
    cudaMalloc(&dA, bytesA);
    cudaMalloc(&dB, bytesB);
    cudaMalloc(&dC, bytesC);

    // Copy operands
    cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice);

    // Launch Config
    constexpr int T = 32;
    dim3 threads(T, T);
    dim3 blocks((n + T - 1) / T, (m + T - 1) / T);

    // Launch kernel
    tiled_matmul<T><<<blocks, threads>>>(dA, dB, dC, m, n, k);

    // Synchronize 
    cudaDeviceSynchronize();

    // Copy dC to hC
    cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost);

    // Reference matmul
    float* hC_ref = matmul_cpu(hA, hB, m, n, k);

    // Check correctness
    check_correctness(hC, hC_ref, m*n);

    // Freee up the space
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);

    return 0;

}