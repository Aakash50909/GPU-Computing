import numpy as np
import time

# Load data from Assignment 5 file, reshape into matrix
data = np.loadtxt("random_data.txt")
size = 1000  # 1000x1000 matrix
A = data[:size*size].reshape(size, size)
B = data[:size*size].reshape(size, size)  # reuse same data

# --- MATRIX ADDITION ---
start = time.time()
C_add = A + B
end = time.time()
print(f"Matrix Addition Time (CPU): {end - start:.4f}s")

# --- MATRIX MULTIPLICATION ---
start = time.time()
C_mul = A @ B   # @ is matrix multiply in numpy
end = time.time()
print(f"Matrix Multiplication Time (CPU): {end - start:.4f}s")

import cupy as cp
import time

size = 1000
A_gpu = cp.random.rand(size, size)
B_gpu = cp.random.rand(size, size)

# --- GPU ADDITION ---
cp.cuda.Stream.null.synchronize()   # wait for GPU to be ready
start = time.time()
C_add = A_gpu + B_gpu
cp.cuda.Stream.null.synchronize()   # wait for GPU to finish
end = time.time()
print(f"Matrix Addition Time (GPU): {end - start:.4f}s")

# --- GPU MULTIPLICATION ---
cp.cuda.Stream.null.synchronize()
start = time.time()
C_mul = A_gpu @ B_gpu
cp.cuda.Stream.null.synchronize()
end = time.time()
print(f"Matrix Multiplication Time (GPU): {end - start:.4f}s")
