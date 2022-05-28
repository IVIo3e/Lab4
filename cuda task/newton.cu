#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include <omp.h>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#define N 100000
#define EPS 1e-3

#define LEFT 0.0f
#define RIGHT 100.0f

double func(double x);
double dfunc(double x);

__device__ double f(double x) {
	return tan(1.262 * x) - 1.84 * x;
}

__device__ double df(double x) {
	return 1.262 / pow(cos(1.262 * x), 2) - 1.84;
}

template<class T> struct Newthon {
	T step;
	Newthon(T _step) {
		step = _step;
	}
	__device__ T operator()(T& x) const {

		double h = f(x) / df(x);
		int count = 0;

		while (abs(h) >= EPS && count++ <= 100)
		{
			h = f(x) / df(x);

			x = x - h;
		}

		return x;
	}
};

int main()
{
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float step = fabs(LEFT - RIGHT) / N;
	printf("Solution search range %.1f  %.1f\nSearch Step %.5f\n", LEFT, RIGHT, step);

	thrust::host_vector<double> data(N);
	thrust::sequence(data.begin(), data.end(), LEFT, step);

	thrust::device_vector<double> input = data;
	thrust::device_vector<double> output(input.size());

	thrust::host_vector<double> result_gpu(input.size());
	thrust::host_vector<double> result_cpu(0);

	Newthon<double> f(step);

	cudaEventRecord(start, 0);

	thrust::transform(input.begin(), input.end(), output.begin(), f);
	thrust::copy(output.begin(), output.end(), result_gpu.begin());

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	for (size_t i = 0; i < result_gpu.size(); i++) {
		if (abs(func(result_gpu[i])) <= EPS)
			printf("%.10f\n", result_gpu[i]);
	}
	printf("\nGPU time: %.10f milliseconds\n", gpuTime);
	double start_cpu;
	double stop_cpu;
	printf("\n\n\n");
	start_cpu = omp_get_wtime();

	for (float element : data) {
		float x = element;
		float h = func(x) / dfunc(x);
		int count = 0;

		while (abs(h) >= EPS && count++ <= 100)
		{
			h = func(x) / dfunc(x);

			x = x - h;
		}

		if (abs(func(x)) <= EPS)
			result_cpu.push_back(x);
	}

	stop_cpu = omp_get_wtime();

	for (float element : result_cpu) {
		printf("%.10f\n", element);
	}

	printf("\nGPU time: %.10f milliseconds\n", gpuTime);
	printf("CPU time %.10f milliseconds\n", (stop_cpu - start_cpu) * 1000);
	return 0;
}

double func(double x) {
	return tan(1.262 * x) - 1.84 * x;
}

double dfunc(double x) {
	return 1.262 / pow(cos(1.262 * x), 2) - 1.84;
}
