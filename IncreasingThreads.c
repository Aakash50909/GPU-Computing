#include <stdio.h>
#include <time.h>

#define N 1024   // matrix size NxN

// Kernel: each thread computes one element of C = A + B
__global__ void matAdd(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // which row am I?
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // which col am I?

    if (row < n && col < n) {
        C[row * n + col] = A[row * n + col] + B[row * n + col];
    }
}

int main() {
    int size = N * N * sizeof(float);

    float *A, *B, *C;           // CPU pointers
    float *dA, *dB, *dC;        // GPU pointers (d = device)

    // Allocate on CPU
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    // Fill with dummy data
    for (int i = 0; i < N*N; i++) { A[i] = 1.0f; B[i] = 2.0f; }

    // Allocate on GPU
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    // Copy CPU → GPU
    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    // Try different thread counts: 8, 16, 32
    int threads = 16;
    dim3 blockSize(threads, threads);                          // threads x threads per block
    dim3 gridSize((N + threads-1)/threads, (N + threads-1)/threads);  // enough blocks to cover N

    clock_t start = clock();
    matAdd<<<gridSize, blockSize>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();    // wait for GPU to finish
    clock_t end = clock();

    printf("Threads per block: %dx%d | Time: %.4f ms\n",
           threads, threads, 1000.0*(end-start)/CLOCKS_PER_SEC);

    // Copy result back
    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B); free(C);
    return 0;
}
