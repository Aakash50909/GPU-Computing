#include <stdio.h>
#define N 1024
#define MASK_SIZE 5
#define TILE 256

#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

__global__ void conv1D(float* input, float* mask, float* output, int n, int maskSize) {
    __shared__ float tile[TILE + MASK_SIZE - 1];  // extra space for halo elements

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halo = maskSize / 2;  // how many elements to peek left/right

    // Load main elements
    tile[threadIdx.x + halo] = (idx < n) ? input[idx] : 0.0f;

    // Load left halo (boundary check)
    if (threadIdx.x < halo) {
        int leftIdx = idx - halo;
        tile[threadIdx.x] = (leftIdx >= 0) ? input[leftIdx] : 0.0f;
    }

    // Load right halo (boundary check)
    if (threadIdx.x >= blockDim.x - halo) {
        int rightIdx = idx + blockDim.x;
        tile[threadIdx.x + halo + blockDim.x - (blockDim.x - halo)] =
            (rightIdx < n) ? input[rightIdx] : 0.0f;
    }

    __syncthreads();

    // Compute convolution
    if (idx < n) {
        float sum = 0;
        for (int k = 0; k < maskSize; k++)
            sum += tile[threadIdx.x + k] * mask[k];
        output[idx] = sum;
    }
}

int main() {
    float *input, *output, *mask;
    float *dIn, *dOut, *dMask;

    input  = (float*)malloc(N * sizeof(float));
    output = (float*)malloc(N * sizeof(float));
    mask   = (float*)malloc(MASK_SIZE * sizeof(float));

    for (int i = 0; i < N; i++) input[i] = (float)i;
    for (int i = 0; i < MASK_SIZE; i++) mask[i] = 1.0f / MASK_SIZE;  // averaging filter

    CHECK(cudaMalloc(&dIn,   N * sizeof(float)));
    CHECK(cudaMalloc(&dOut,  N * sizeof(float)));
    CHECK(cudaMalloc(&dMask, MASK_SIZE * sizeof(float)));

    CHECK(cudaMemcpy(dIn,   input, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dMask, mask,  MASK_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    conv1D<<<(N + TILE-1)/TILE, TILE>>>(dIn, dMask, dOut, N, MASK_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("1D Conv Time: %.3f ms\n", ms);

    CHECK(cudaMemcpy(output, dOut, N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("output[5] = %.3f\n", output[5]);

    cudaFree(dIn); cudaFree(dOut); cudaFree(dMask);
    free(input); free(output); free(mask);
    return 0;
}
