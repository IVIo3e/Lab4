#include <iostream>  
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define	N (512*512)		

__global__ void kernel(float* data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    data[idx] = idx * idx;
}

int main(int argc, char* argv[])
{
    float   a[N];
    float* dev = NULL;

    cudaMalloc((void**)&dev, N * sizeof(float));

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    kernel << <dim3((N / 512), 1), dim3(512, 1) >> > (dev);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("GPU : %.10f ms\n\n", gpuTime);

    cudaMemcpy(a, dev, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev);


    cudaEvent_t start_, stop_;
    float gpuTime_ = 0.0f;
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_, 0);

    for (int i = 0; i < N; i++)
    {
        a[i] = i * i;
    }
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&gpuTime_, start_, stop_);
    printf("CPU : %.10f ms\n\n", gpuTime_);

    return 0;
}
