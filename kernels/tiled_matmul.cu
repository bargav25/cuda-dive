#include <stdio.h>
#include <stdlib.h>

template<int T>
__global__ void tiled_matmul(float *A, float *B, float *C, int M, int N, int K) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;


    // __shared__ to declare variable in shared memory so that data is shared between the threads in a block
    // prevents from each thread accessing items from global memory again and again
    __shared__ float sA[T][T]; // Tile A: rows of A, cols of K
    __shared__ float sB[T][T]; // Tile B: rows of K, cols of B

    float acc = 0.0f; // Same as naive gemm, each thread writing the output of a single cell

    // Iterating over K dimensions in tiles of size 
    int num_tiles = (K + T - 1) / T;

    for (int t=0; t < num_tiles; t++) { // for each tile

        // all threads in block collectively load A[row, t*T: (t+1)*T]
        // all threads in block collectively load B[t*T: (t+1)*T, col]

        int a_col = t * T + threadIdx.x; // K index for A
        int b_row = t * T + threadIdx.y; // K index for B

        // Load A tile (with boundary check) 
        if (row < M && a_col < K) sA[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else                    sA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile (with boundary check)
        if (b_row < K && col < N) sB[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else                    sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); // make sure everyone finished loading

        // Multiply the two tiles
        // each thread accumulates its dot product using sA and sB
        for (int k = 0; k < T; k++) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads(); // before next tile overwrites shared mem


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
    int m = 1017, n = 213, k = 1451;

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