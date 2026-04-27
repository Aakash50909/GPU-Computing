#include <stdio.h>
#include <curand.h>

#define N 1024
#define TILE 16

// Error check macro — wrap every CUDA call with this
#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

__global__ void tiledMatMul(float* A, float* B, float* C, int n) {
    // Shared memory tiles — live in fast on-chip memory
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    // Slide the tile across the row/col
    for (int t = 0; t < n / TILE; t++) {
        tileA[threadIdx.y][threadIdx.x] = A[row * n + (t * TILE + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];
        __syncthreads();    // wait — all threads must finish loading before computing

        for (int k = 0; k < TILE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();    // wait before loading next tile
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

int main() {
    int size = N * N * sizeof(float);
    float *dA, *dB, *dC, *C;

    CHECK(cudaMalloc(&dA, size));
    CHECK(cudaMalloc(&dB, size));
    CHECK(cudaMalloc(&dC, size));

    // Generate random data using cuRAND
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, dA, N * N);
    curandGenerateUniform(gen, dB, N * N);
    curandDestroyGenerator(gen);

    // Timing using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    cudaEventRecord(start);
    tiledMatMul<<<grid, block>>>(dA, dB, dC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Tiled MatMul Time: %.3f ms\n", ms);

    C = (float*)malloc(size);
    CHECK(cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost));

    cudaFree(dA); cudaFree(dB); cudaFree(dC); free(C);
    return 0;
}
