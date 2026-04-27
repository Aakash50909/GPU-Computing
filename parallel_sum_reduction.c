#include <stdio.h>
#define N 1048576   // 2^20 elements
#define BLOCK 256

#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

__global__ void sumReduce(float* input, float* output, int n) {
    __shared__ float tile[BLOCK];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory (0 if out of bounds)
    tile[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Tree reduction — halve active threads each step
    for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
        if (tid < stride)
            tile[tid] += tile[tid + stride];   // lower half adds upper half
        __syncthreads();
    }

    // Thread 0 of each block writes its block's sum
    if (tid == 0)
        output[blockIdx.x] = tile[0];
}

int main() {
    float *A, *dA, *dOut, *out;
    int gridSize = (N + BLOCK - 1) / BLOCK;

    A = (float*)malloc(N * sizeof(float));
    out = (float*)malloc(gridSize * sizeof(float));
    for (int i = 0; i < N; i++) A[i] = 1.0f;  // sum should = N

    CHECK(cudaMalloc(&dA, N * sizeof(float)));
    CHECK(cudaMalloc(&dOut, gridSize * sizeof(float)));
    CHECK(cudaMemcpy(dA, A, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    sumReduce<<<gridSize, BLOCK>>>(dA, dOut, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    CHECK(cudaMemcpy(out, dOut, gridSize * sizeof(float), cudaMemcpyDeviceToHost));

    float total = 0;
    for (int i = 0; i < gridSize; i++) total += out[i];  // final sum on CPU

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Sum = %.0f | Time: %.3f ms\n", total, ms);

    cudaFree(dA); cudaFree(dOut); free(A); free(out);
    return 0;
}
