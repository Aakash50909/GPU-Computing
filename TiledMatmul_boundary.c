#include <stdio.h>
#define N 1000    // intentionally NOT a multiple of 16 to test boundary
#define TILE 16

#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

__global__ void safeTiledMatMul(float* A, float* B, float* C, int n) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    int numTiles = (n + TILE - 1) / TILE;  // ceiling division

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        // Boundary check before loading — load 0 if out of bounds
        tileA[threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? A[row * n + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < n && col < n) ? B[bRow * n + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }

    // Boundary check before writing result
    if (row < n && col < n)
        C[row * n + col] = sum;
}

int main() {
    int size = N * N * sizeof(float);
    float *A, *B, *C, *dA, *dB, *dC;

    A = (float*)malloc(size); B = (float*)malloc(size); C = (float*)malloc(size);
    for (int i = 0; i < N*N; i++) { A[i] = 1.0f; B[i] = 2.0f; }

    CHECK(cudaMalloc(&dA, size)); CHECK(cudaMalloc(&dB, size)); CHECK(cudaMalloc(&dC, size));
    CHECK(cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE-1)/TILE, (N + TILE-1)/TILE);   // ceiling grid size

    cudaEventRecord(start);
    safeTiledMatMul<<<grid, block>>>(dA, dB, dC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Safe Tiled MatMul Time: %.3f ms\n", ms);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B); free(C);
    return 0;
}
