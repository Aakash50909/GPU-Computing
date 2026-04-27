#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define MAX 100

// GPU kernel — each thread gets its own random state
__global__ void makeRandom(int* result) {
    curandState_t state;                    // each thread needs its own state

    curand_init(1234,   // seed  — starting point of random sequence
                0,      // sequence — use threadIdx.x if multiple threads
                0,      // offset — skip ahead by this much (usually 0)
                &state);

    *result = curand(&state) % MAX;         // get random number, limit to MAX
}

int main() {
    int* gpu_num;
    cudaMalloc(&gpu_num, sizeof(int));      // allocate 1 int on GPU

    makeRandom<<<1, 1>>>(gpu_num);          // 1 block, 1 thread

    int result;
    cudaMemcpy(&result, gpu_num, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Random number = %d\n", result);
    cudaFree(gpu_num);
    return 0;
}
