# CUDA
nvcc program.cu -o program -lcurand

# OpenACC
nvc -acc -Minfo=accel program.c -o program

# OpenCL
gcc program.c -o program -lOpenCL
