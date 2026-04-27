#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>

#define N 4096

void fillRandom(float* arr, int n) {
    srand(123);
    for (int i = 0; i < n; i++)
        arr[i] = (float)rand() / RAND_MAX;
}

void printFirst5(float* arr) {
    for (int i = 0; i < 5; i++) printf("%.4f ", arr[i]);
    printf("\n");
}

// ---- BUBBLE SORT ----
void bubbleSort(float* arr, int n) {
    clock_t start = clock();

    for (int pass = 0; pass < n; pass++) {
        #pragma acc parallel loop copyin(arr[0:n]) copyout(arr[0:n])
        for (int i = 0; i < n - 1; i++) {
            if (arr[i] > arr[i + 1]) {
                float tmp = arr[i];
                arr[i]   = arr[i + 1];
                arr[i + 1] = tmp;
            }
        }
    }

    clock_t end = clock();
    printf("Bubble Sort Time: %.4f sec\n", (double)(end-start)/CLOCKS_PER_SEC);
}

// ---- ODD-EVEN SORT ----
void oddEvenSort(float* arr, int n) {
    clock_t start = clock();

    for (int pass = 0; pass < n; pass++) {
        // Even phase
        #pragma acc parallel loop
        for (int i = 0; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                float tmp = arr[i]; arr[i] = arr[i+1]; arr[i+1] = tmp;
            }
        }
        // Odd phase
        #pragma acc parallel loop
        for (int i = 1; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                float tmp = arr[i]; arr[i] = arr[i+1]; arr[i+1] = tmp;
            }
        }
    }

    clock_t end = clock();
    printf("Odd-Even Sort Time: %.4f sec\n", (double)(end-start)/CLOCKS_PER_SEC);
}

// ---- BITONIC SORT ----
// Works on arrays of size = power of 2
// direction: 1 = ascending, 0 = descending
void bitonicCompare(float* arr, int i, int j, int dir) {
    if (dir == (arr[i] > arr[j])) {
        float tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

void bitonicSort(float* arr, int n) {
    clock_t start = clock();

    // k = size of bitonic sequence being merged
    for (int k = 2; k <= n; k *= 2) {
        // j = stride within the sequence
        for (int j = k / 2; j > 0; j /= 2) {
            #pragma acc parallel loop
            for (int i = 0; i < n; i++) {
                int partner = i ^ j;           // XOR gives the compare partner
                if (partner > i) {
                    int dir = (i & k) == 0;    // ascending or descending block?
                    bitonicCompare(arr, i, partner, dir);
                }
            }
        }
    }

    clock_t end = clock();
    printf("Bitonic Sort Time: %.4f sec\n", (double)(end-start)/CLOCKS_PER_SEC);
}

int main() {
    float* arr = (float*)malloc(N * sizeof(float));

    printf("=== Bubble Sort ===\n");
    fillRandom(arr, N);
    bubbleSort(arr, N);
    printFirst5(arr);

    printf("=== Odd-Even Sort ===\n");
    fillRandom(arr, N);
    oddEvenSort(arr, N);
    printFirst5(arr);

    printf("=== Bitonic Sort ===\n");
    fillRandom(arr, N);
    bitonicSort(arr, N);
    printFirst5(arr);

    free(arr);
    return 0;
}
