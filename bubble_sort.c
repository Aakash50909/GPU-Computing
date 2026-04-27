#include <stdio.h>
#include <curand.h>
#define N 1024
#define BLOCK 256

#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// Each block sorts its own chunk in shared memory
__global__ void bubbleSort(float* data, int n) {
    __shared__ float tile[BLOCK];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n) tile[tid] = data[idx];
    __syncthreads();

    // Bubble sort within the tile
    for (int i = 0; i < BLOCK; i++) {
        for (int j = tid; j < BLOCK - 1; j += 1) {
            if (tile[j] > tile[j + 1]) {
                float tmp = tile[j];
                tile[j] = tile[j + 1];
                tile[j + 1] = tmp;
            }
            __syncthreads();
        }
    }

    if (idx < n) data[idx] = tile[tid];
}

int main() {
    float *dData, *data;
    data = (float*)malloc(N * sizeof(float));

    CHECK(cudaMalloc(&dData, N * sizeof(float)));

    // Generate random data with cuRAND
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42ULL);
    curandGenerateUniform(gen, dData, N);
    curandDestroyGenerator(gen);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    bubbleSort<<<(N + BLOCK-1)/BLOCK, BLOCK>>>(dData, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Bubble Sort Time: %.3f ms\n", ms);

    CHECK(cudaMemcpy(data, dData, N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("First 5 sorted: %.3f %.3f %.3f %.3f %.3f\n",
           data[0], data[1], data[2], data[3], data[4]);

    cudaFree(dData); free(data);
    return 0;
}
