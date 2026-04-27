# On Colab — install NVIDIA HPC SDK (has nvc compiler with OpenACC)
!apt-get install -y nvidia-hpc-sdk
# Or use pgcc/nvc which comes with it
# Compile command for all programs below:
# nvc -acc -Minfo=accel program.c -o program
// Q2 — Display OpenACC info
#include <stdio.h>
#include <openacc.h>   // OpenACC header

int main() {
    // How many GPUs available?
    int numDevices = acc_get_num_devices(acc_device_nvidia);
    printf("Number of GPUs: %d\n", numDevices);

    // Which device type is active?
    acc_device_t devType = acc_get_device_type();
    printf("Device type: %d\n", devType);   // 5 = NVIDIA GPU

    // Current device number
    printf("Current device: %d\n", acc_get_device_num(acc_device_nvidia));

    return 0;
}
