#include <stdio.h>
#include <openacc.h>

#define N 1000000

int main() {
    float a[N], b[N], c_add[N], c_mul[N];

    // Initialize arrays
    for (int i = 0; i < N; i++) { a[i] = 2.0f; b[i] = 3.0f; }

    // ---- METHOD 1: parallel loop (you control parallelism) ----
    #pragma acc parallel loop
    for (int i = 0; i < N; i++) {
        c_add[i] = a[i] + b[i];   // addition
    }

    // ---- METHOD 2: kernels (compiler decides parallelism) ----
    #pragma acc kernels
    for (int i = 0; i < N; i++) {
        c_mul[i] = a[i] * b[i];   // multiplication
    }

    printf("Addition   a[0]+b[0] = %.1f\n", c_add[0]);   // 5.0
    printf("Multiply   a[0]*b[0] = %.1f\n", c_mul[0]);   // 6.0

    return 0;
}
