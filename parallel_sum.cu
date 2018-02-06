/*
	Author: Yajie Zhao
	This cu file compares two methods (a naive one and parallel reduction) that calculate the sum
	of an array. 
*/
#include <stdio.h>
#include <stdlib.h>
#include "timerc.h"

#define N 1024*1024
#define THREADS_PER_BLOCK 1024
#define MULT 32

__global__ void parallel_sum_naive(int *a, int *b, int mult) {
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int sum = 0;
	for (int i = 0; i < mult; i++) {
		sum += a[ix * mult + i];
	}
	b[ix] = sum;
}

// parallel reduction
__global__ void parallel_sum_reduction(int *a, int *b, int n, int block_size) {
	int start = blockIdx.x * block_size;
	int step = 1;

	while (step <= block_size/2) {
		if (threadIdx.x < (block_size / step / 2)) {
			a[start + 2 * threadIdx.x * step] += a[start + 2 * threadIdx.x * step + step];
		}
		__syncthreads();
		step *= 2;
	}
	if (threadIdx.x == 0) b[blockIdx.x] = a[start];
}



int main() {
	cudaSetDevice(0);
	// set environment
	int num_blocks = N / THREADS_PER_BLOCK / MULT;
	float cpuTime, gpuSumTime, gpuSetupTime, gpuTotalTime;
	int *dev_arr, *dev_output;
	int *host_arr = (int *) malloc(N * sizeof(int));
	int *host_output = (int *) malloc(N / MULT * sizeof(int));
	for (int i = 0; i < N; i++) {
		host_arr[i] = i;
	}
	long long int expectedSum = N / 2 * (long long int) (N-1), cpuSum, gpuSum;

	// cpu time to calculate sum
	cstart();
	cpuSum = 0;
	for (int i = 0; i < N; i++) {
		cpuSum += host_arr[i];
	}
	cend(&cpuTime);
	printf("cpu time = %f\n", cpuTime);

	// gpu time to calculate sum using naive method
	gpuTotalTime = 0.0;
	gstart();
	gpuSum = 0;
	cudaMalloc((void **) &dev_arr, N * sizeof(int));
	cudaMalloc((void **) &dev_output, N/MULT * sizeof(int));
	cudaMemcpy(dev_arr, host_arr, N * sizeof(int), cudaMemcpyHostToDevice);
	gend(&gpuSetupTime);
	gpuTotalTime += gpuSetupTime;

	gstart();
	parallel_sum_naive<<<num_blocks, THREADS_PER_BLOCK>>>(dev_arr, dev_output, MULT);
	gend(&gpuSumTime);
	gpuTotalTime += gpuSumTime;

	gstart();
	cudaMemcpy(host_output, dev_output, N / MULT * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N / MULT; i++) {
		gpuSum += host_output[i];
	}
	gend(&gpuSetupTime);
	gpuTotalTime += gpuSetupTime;
	printf("gpu sum time (with naive) = %f, gpu total time = %f\n", gpuSumTime, gpuTotalTime);
	gpuTotalTime -= (gpuSumTime + gpuSetupTime);

	// set environment for second method
	num_blocks *= MULT;
	free(host_output);
	host_output = (int *) malloc(num_blocks * sizeof(int));
	cudaFree(dev_output);
	cudaMalloc((void **) &dev_output, num_blocks * sizeof(int));

	// gpu time to calculate sum using parallel reduction
	gstart();
	parallel_sum_reduction<<<num_blocks, THREADS_PER_BLOCK>>>(dev_arr, dev_output, N, THREADS_PER_BLOCK);
	gend(&gpuSumTime);
	gpuTotalTime += gpuSumTime;

	gstart();
	gpuSum = 0;
	cudaMemcpy(host_output, dev_output, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < num_blocks; i++) {
		gpuSum += host_output[i];
	}
	gend(&gpuSetupTime);
	gpuTotalTime += gpuSetupTime;
	printf("gpu sum time (with reduction) = %f, gpu total time = %f\n", gpuSumTime, gpuTotalTime);

	printf("Expected Sum is %lld, cpu Sum is %lld, gpu Sum is %lld\n", expectedSum, cpuSum, gpuSum);
	free(host_arr);
	free(host_output);
	cudaFree(dev_arr);
	cudaFree(dev_output);
}