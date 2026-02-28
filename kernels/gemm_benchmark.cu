#include <stdio.h>
#include <functional>



__global__ void naive_gemm(float *A, float *B, float *C, int M, int N, int K) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= N || row >= M) return;

    float acc = 0.0f;

    // C[row, col] = sum_k A[row, k] * B[k, col]
    for (int k=0; k<K; k++) acc += A[row * K + k] * B[k * N + col];

    C[row * N + col] = acc;

}

template<int T>
__global__ void tiled_gemm(float *A, float *B, float *C, int M, int N, int K) {

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

template<int T, int Ts> // (T, T): Number of threads per block,  (Ts, Ts): Number of cells each thread owns (or accumulates)
__global__ void regtiled_gemm(float *A, float *B, float* C, int M, int N, int K) {

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


float time_kernel(std::function<void()> kernel_launch, int warmup=3, int n_iters=10) {

    for (int i=0; i < warmup; i++) kernel_launch();

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Pushing the work to Cuda Stream
    cudaEventRecord(start); // Async Event
    for (int i=0; i < n_iters; i++) kernel_launch(); // Async 
    cudaEventRecord(stop); // Async Event

    cudaEventSynchronize(stop); // Block the CPU unti the GPU reaches the stop event

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / n_iters;

}

void initialize_arrs(float *A, float *B, int m, int n, int k) {

    // Filling inputs with determinsitic values
    for (int i = 0; i < m*k; i++) A[i] = 0.01 * float(i % 97);
    for (int i = 0; i < n*k; i++) B[i] = 0.02 * float(i % 89);

}

int main() {

    // --- Correctness check on small matrix first ---
    {
        int M = 64, N = 64, K = 64;
        size_t bA = M*K*sizeof(float), bB = K*N*sizeof(float), bC = M*N*sizeof(float);

        float *hA = (float*)malloc(bA), *hB = (float*)malloc(bB), *hC = (float*)malloc(bC);
        initialize_arrs(hA, hB, M, N, K);
        float *hC_ref = matmul_cpu(hA, hB, M, N, K);

        float *dA, *dB, *dC;
        cudaMalloc(&dA, bA); cudaMalloc(&dB, bB); cudaMalloc(&dC, bC);
        cudaMemcpy(dA, hA, bA, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, bB, cudaMemcpyHostToDevice);

        constexpr int T_regtiled = 16, Ts_regtiled = 4, Bs = T_regtiled * Ts_regtiled;
        dim3 threads_naive(32, 32);
        dim3 threads_tiled(32, 32);
        dim3 threads_regtiled(T_regtiled, T_regtiled);
        dim3 blocks_naive((N+31)/32, (M+31)/32);
        dim3 blocks_tiled((N+31)/32, (M+31)/32);
        dim3 blocks_regtiled((N+Bs-1)/Bs, (M+Bs-1)/Bs);

        auto verify = [&](const char* name, std::function<void()> fn) {
            cudaMemset(dC, 0, bC);
            fn();
            cudaDeviceSynchronize();
            cudaMemcpy(hC, dC, bC, cudaMemcpyDeviceToHost);
            float max_err = 0.0f;
            for (int i = 0; i < M*N; i++) max_err = fmaxf(max_err, fabsf(hC[i] - hC_ref[i]));
            printf("correctness %-20s max_err=%.6f  %s\n", name, max_err, max_err < 0.01f ? "OK" : "WRONG");
        };

        verify("naive",    [&]{ naive_gemm<<<blocks_naive, threads_naive>>>(dA, dB, dC, M, N, K); });
        verify("tiled",    [&]{ tiled_gemm<32><<<blocks_tiled, threads_tiled>>>(dA, dB, dC, M, N, K); });
        verify("regtiled", [&]{ regtiled_gemm<T_regtiled, Ts_regtiled><<<blocks_regtiled, threads_regtiled>>>(dA, dB, dC, M, N, K); });

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        free(hA); free(hB); free(hC); free(hC_ref);
        printf("\n");
    }

    // --- Benchmark on large matrix ---
    {
        int M = 4096, N = 4096, K = 4096;
        double flops = 2.0 * M * N * K;

        size_t bA = M*K*sizeof(float), bB = K*N*sizeof(float), bC = M*N*sizeof(float);

        float *hA = (float*)malloc(bA), *hB = (float*)malloc(bB), *hC = (float*)malloc(bC);
        initialize_arrs(hA, hB, M, N, K);

        float *dA, *dB, *dC;
        cudaMalloc(&dA, bA); cudaMalloc(&dB, bB); cudaMalloc(&dC, bC);
        cudaMemcpy(dA, hA, bA, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, bB, cudaMemcpyHostToDevice);

        constexpr int T_regtiled = 16, Ts_regtiled = 4, Bs = T_regtiled * Ts_regtiled;

        dim3 threads_naive(32, 32);
        dim3 threads_tiled(32, 32);
        dim3 threads_regtiled(T_regtiled, T_regtiled);

        dim3 blocks_naive((N+31)/32,       (M+31)/32);
        dim3 blocks_tiled((N+31)/32,       (M+31)/32);
        dim3 blocks_regtiled((N+Bs-1)/Bs,  (M+Bs-1)/Bs);

        float ms_naive    = time_kernel([&]{ naive_gemm<<<blocks_naive, threads_naive>>>(dA, dB, dC, M, N, K); });
        float ms_tiled    = time_kernel([&]{ tiled_gemm<32><<<blocks_tiled, threads_tiled>>>(dA, dB, dC, M, N, K); });
        float ms_regtiled = time_kernel([&]{ regtiled_gemm<T_regtiled, Ts_regtiled><<<blocks_regtiled, threads_regtiled>>>(dA, dB, dC, M, N, K); });

        printf("%-20s %8.3f ms  %8.1f GFLOPS  (%.1f%% of peak)\n", "naive",    ms_naive,    flops/(ms_naive*1e6),    flops/(ms_naive*1e6)/91600*100);
        printf("%-20s %8.3f ms  %8.1f GFLOPS  (%.1f%% of peak)\n", "tiled",    ms_tiled,    flops/(ms_tiled*1e6),    flops/(ms_tiled*1e6)/91600*100);
        printf("%-20s %8.3f ms  %8.1f GFLOPS  (%.1f%% of peak)\n", "regtiled", ms_regtiled, flops/(ms_regtiled*1e6), flops/(ms_regtiled*1e6)/91600*100);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        free(hA); free(hB); free(hC);
    }

    return 0;
}