import numpy as np
import time

# Generate 1 million random numbers
n = 1_000_000

start = time.time()                          # start the stopwatch
data = np.random.rand(n)                     # generate random floats [0, 1)
end = time.time()                            # stop the stopwatch

# Save to file
np.savetxt("random_data.txt", data)

print(f"Generated {n} numbers in {end - start:.4f} seconds")

# Run these in separate Colab cells

# Check if GPU is available
!nvidia-smi

# Install Numba (JIT compiler with CUDA support)
!pip install numba

# Install CuPy (NumPy but on GPU)
!pip install cupy-cuda12x   # use cupy-cuda11x if CUDA 11

# Verify
import numba
import cupy as cp
print("Numba version:", numba.__version__)
print("CuPy version:", cp.__version__)

# Quick CuPy test
a = cp.array([1, 2, 3])
print("CuPy array on GPU:", a)

from numba import cuda

# Get the GPU device
gpu = cuda.get_current_device()

print("GPU Name              :", gpu.name)
print("Max Threads per Block :", gpu.MAX_THREADS_PER_BLOCK)
print("Max Block Dimensions  :", gpu.MAX_BLOCK_DIM_X, gpu.MAX_BLOCK_DIM_Y, gpu.MAX_BLOCK_DIM_Z)
print("Max Grid Dimensions   :", gpu.MAX_GRID_DIM_X, gpu.MAX_GRID_DIM_Y, gpu.MAX_GRID_DIM_Z)
print("Warp Size             :", gpu.WARP_SIZE)
print("Compute Capability    :", gpu.compute_capability)

import cupy as cp

dev = cp.cuda.Device(0)
props = cp.cuda.runtime.getDeviceProperties(0)
print("GPU Name:", props['name'])
print("Total Memory (GB):", props['totalGlobalMem'] / 1e9)
print("Multiprocessors:", props['multiProcessorCount'])
print("Clock Rate (GHz):", props['clockRate'] / 1e6)
