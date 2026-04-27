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
