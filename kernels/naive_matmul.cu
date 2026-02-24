#include <stdio.h>
#include <stdlib.h>

__global__ void naive_matmul(float *A, float *B, float *C, int M, int N, int K) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= N || row >= M) return;

    float acc = 0.0f;

    // C[row, col] = sum_k A[row, k] * B[k, col]
    for (int k=0; k<K; k++) acc += A[row * K + k] * B[k * N + col];

    C[row * N + col] = acc;

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
    int m = 17, n = 13, k = 11;

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
    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);

    // Launch kernel
    naive_matmul<<<blocks, threads>>>(dA, dB, dC, m, n, k);

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