#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>

#define N 1024

// OpenACC doesn't have built-in error macros like CUDA
// Best practice: check device availability before running
void checkDevice() {
    if (acc_get_num_devices(acc_device_nvidia) == 0) {
        printf("ERROR: No GPU found! Exiting.\n");
        exit(1);
    }
    printf("GPU found. Running on device %d\n",
           acc_get_device_num(acc_device_nvidia));
}

int main() {
    checkDevice();

    // Allocate matrices
    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(N * N * sizeof(float));
    float *C = (float*)malloc(N * N * sizeof(float));

    // Initialize with random data
    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    // ---- MATRIX ADDITION ----
    clock_t start = clock();

    #pragma acc parallel loop collapse(2) copyin(A[0:N*N], B[0:N*N]) copyout(C[0:N*N])
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i * N + j] = A[i * N + j] + B[i * N + j];

    clock_t end = clock();
    printf("Matrix Addition Time: %.4f sec\n", (double)(end-start)/CLOCKS_PER_SEC);

    // ---- MATRIX MULTIPLICATION ----
    start = clock();

    #pragma acc parallel loop collapse(2) copyin(A[0:N*N], B[0:N*N]) copyout(C[0:N*N])
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }

    end = clock();
    printf("Matrix Multiply Time: %.4f sec\n", (double)(end-start)/CLOCKS_PER_SEC);

    free(A); free(B); free(C);
    return 0;
}
