#include <stdio.h>
#define N 1024
#define TILE 16
#define COARSE 2   // each thread handles COARSE elements in a row

#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

__global__ void coarseMatMul(float* A, float* B, float* C, int n) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int colStart = blockIdx.x * TILE * COARSE + threadIdx.x;  // first column this thread handles

    float sums[COARSE] = {0};   // accumulate COARSE results

    for (int t = 0; t < n / TILE; t++) {
        tileA[threadIdx.y][threadIdx.x] = A[row * n + t * TILE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + colStart];
        __syncthreads();

        for (int k = 0; k < TILE; k++)
            for (int c = 0; c < COARSE; c++)
                sums[c] += tileA[threadIdx.y][k] * tileB[k][threadIdx.x + c * TILE];
        __syncthreads();
    }

    for (int c = 0; c < COARSE; c++)
        if (row < n && colStart + c * TILE < n)
            C[row * n + colStart + c * TILE] = sums[c];
}

int main() {
    int size = N * N * sizeof(float);
    float *A, *B, *C, *dA, *dB, *dC;

    A = (float*)malloc(size); B = (float*)malloc(size); C = (float*)malloc(size);
    for (int i = 0; i < N*N; i++) { A[i] = 1.0f; B[i] = 1.0f; }

    CHECK(cudaMalloc(&dA, size)); CHECK(cudaMalloc(&dB, size)); CHECK(cudaMalloc(&dC, size));
    CHECK(cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    dim3 block(TILE, TILE);
    dim3 grid(N / (TILE * COARSE), N / TILE);

    cudaEventRecord(start);
    coarseMatMul<<<grid, block>>>(dA, dB, dC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Coarsened MatMul Time: %.3f ms\n", ms);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B); free(C);
    return 0;
}
