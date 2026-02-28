#include <stdio.h>

template<int T, int Ts> // (T, T): Number of threads per block,  (Ts, Ts): Number of cells each thread owns (or accumulates)
__global__ void gemm(float *A, float *B, float* C, int M, int N, int K) {

    // Logical offset of thread
    int col_offset = blockIdx.x * blockDim.x * Ts + threadIdx.x * Ts;
    int row_offset = blockIdx.y * blockDim.y * Ts + threadIdx.y * Ts;

    // Number of cells in block: (T * Ts, T * Ts)
    constexpr int Bs = T * Ts;

    // Initialzing in shared memory
    __shared__ float sA[Bs][Bs];
    __shared__ float sB[Bs][Bs];

    // Accumator in register memory of the thread
    float acc[Ts][Ts] = {0.0f};

    // Register mem
    float rA[Ts] = {0.0f};
    float rB[Ts] = {0.0f};

    int num_tiles = (K + Bs - 1) / Bs;

    for (int t = 0; t < num_tiles; t++) {

        // int a_row_offset = row_offset; 
        // int b_col_offset = col_offset;
        int a_col_offset = Bs * t + threadIdx.x * Ts;
        int b_row_offset = Bs * t + threadIdx.y * Ts;

        for (int m = 0; m < Ts; m++) {

            for(int n = 0; n < Ts; n++) {

                if (row_offset + m < M && a_col_offset + n < K) sA[threadIdx.y * Ts + m][threadIdx.x * Ts + n] = A[(row_offset + m) * K + a_col_offset + n];
                else                                                sA[threadIdx.y * Ts + m][threadIdx.x * Ts + n] = 0.0f;

                if (b_row_offset + m < K && col_offset + n < N) sB[threadIdx.y * Ts + m][threadIdx.x * Ts + n] = B[(b_row_offset + m) * N + col_offset + n];
                else                                                sB[threadIdx.y * Ts + m][threadIdx.x * Ts + n] = 0.0f;

            }
        }

        __syncthreads(); // make sure everone finished loading

        // Loop over k in the block

        for (int k = 0; k < Bs; k++) {

            for (int m = 0; m < Ts; m++) rA[m] = sA[threadIdx.y * Ts + m][k];
            for (int n = 0; n < Ts; n++) rB[n] = sB[k][threadIdx.x * Ts + n];

            // Accumulate the outer product

            for (int m = 0; m < Ts; m++) {
                for (int n = 0; n < Ts; n++) 
                    acc[m][n] += rA[m] * rB[n];
            }

        }



        __syncthreads(); // before next tile overwrites shared mem


    }

    for (int m = 0; m < Ts; m++) {

        for (int n = 0; n < Ts; n++) {

            if (row_offset + m < M && col_offset + n < N) C[(row_offset + m) * N + col_offset + n] = acc[m][n];
        }
    } 

     
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
    constexpr int T = 16;
    constexpr int Ts = 4;

    constexpr int Bs = T * Ts;

    dim3 threads(T, T);
    dim3 blocks((n + Bs - 1) / Bs, (m + Bs - 1) / Bs);

    // Launch kernel
    gemm<T, Ts><<<blocks, threads>>>(dA, dB, dC, m, n, k);

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