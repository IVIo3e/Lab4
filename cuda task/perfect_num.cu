#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <device_functions.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <omp.h>
#include <stdio.h>


__constant__ unsigned int arraySize[1];

const int THREADS = 1024;

cudaError_t perfectNumberWithCuda(unsigned int size);
__global__ void addKernel();
__device__ bool isPerfect(unsigned int num, int thread);
__host__ void perfectToNum(unsigned int num);

int main()
{
    unsigned int array_size = 100000;

    printf("N --> %d\n", array_size);
    printf("Threads per block --> %d\n\n", THREADS);

    cudaError_t cudaStatus = perfectNumberWithCuda(array_size);
  

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
  
    double start;
    double stop;

    start = omp_get_wtime();  
    perfectToNum(array_size);
    stop = omp_get_wtime();

    printf("Timing CPU Events %.10f", (stop - start) * 1000);

    return 0;
}

cudaError_t perfectNumberWithCuda(unsigned int size)
{
    cudaEvent_t start, stop;
    cudaError_t cudaStatus;

    float gpuTime = 0.0f;

    cudaStatus = cudaMemcpyToSymbol(arraySize, &size, sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Memcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cudaStatus = cudaSetDevice(0);

    addKernel<<<(size + THREADS - 1) / THREADS, THREADS>>>();

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaEventRecord(stop, 0);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    printf("Timing CUDA Events %.10f\n\n", gpuTime);
 
Error:
   
    return cudaStatus;
}

__global__ void addKernel() {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < arraySize[0])
        if (isPerfect(i, threadIdx.x))
            printf("%d\n", i);
}

__device__ bool isPerfect(unsigned int num, int thread) {
    __shared__ int answ_[THREADS];

    answ_[thread] = 1;
    for (int i = 2; i <= num / 2; i++) {
        if (num % i == 0) {
            answ_[thread] += i;
        }
    }
    return (num == answ_[thread] && num != 1);
}

__host__ void perfectToNum(unsigned int num) {
    for (int i = 2; i <= num; i++)
    {
        unsigned int answ = 1;

        for (int j = 2; j <= i / 2; j++) {
            if (i % j == 0) {
                answ += j;
            }
        }

        if (i == answ)
            printf("%d\n", i);
    }
}
