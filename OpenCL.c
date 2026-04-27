# On Ubuntu/Colab
sudo apt-get install opencl-headers ocl-icd-opencl-dev

# Compile command for all programs below:
gcc program.c -o program -lOpenCL && ./program

  #include <stdio.h>
#include <CL/cl.h>      // OpenCL header

int main() {
    cl_uint numPlatforms;

    // Step 1: Get platforms (NVIDIA, AMD, Intel etc.)
    clGetPlatformIDs(0, NULL, &numPlatforms);
    printf("Number of platforms: %d\n", numPlatforms);

    cl_platform_id platforms[numPlatforms];
    clGetPlatformIDs(numPlatforms, platforms, NULL);

    for (int i = 0; i < numPlatforms; i++) {
        char name[128], vendor[128], version[128];

        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,    128, name,    NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,  128, vendor,  NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 128, version, NULL);

        printf("\nPlatform %d:\n", i);
        printf("  Name   : %s\n", name);
        printf("  Vendor : %s\n", vendor);
        printf("  Version: %s\n", version);

        // Step 2: Get devices on this platform
        cl_uint numDevices;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        cl_device_id devices[numDevices];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

        printf("  Devices: %d\n", numDevices);
        for (int j = 0; j < numDevices; j++) {
            char devName[128];
            cl_ulong memSize;
            cl_uint maxUnits;

            clGetDeviceInfo(devices[j], CL_DEVICE_NAME,                128, devName,   NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE,     sizeof(cl_ulong), &memSize, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,   sizeof(cl_uint),  &maxUnits, NULL);

            printf("    Device %d: %s\n", j, devName);
            printf("    Global Mem: %.2f GB\n", memSize / 1e9);
            printf("    Compute Units: %d\n", maxUnits);
        }
    }
    return 0;
}
