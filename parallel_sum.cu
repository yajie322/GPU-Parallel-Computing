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
#define NUM_BLOCKS N/THREADS_PER_BLOCK
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
__global__ void parallel_sum_reduction(int *a, int *b, int block_size) {
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

__global__ void parallel_sum_reduction_with_shared_mem(int *a, int *b, int block_size) {
	__shared__ int tmpmem[THREADS_PER_BLOCK];
	int step = 1;

	tmpmem[threadIdx.x] = a[threadIdx.x + blockDim.x * blockIdx.x];
	__syncthreads();

	while (step <= block_size/2) {
		if (threadIdx.x < (block_size / step / 2)) {
			tmpmem[2 * threadIdx.x * step] += tmpmem[2 * threadIdx.x * step + step];
		}
		__syncthreads();
		step *= 2;
	}
	if (threadIdx.x == 0) b[blockIdx.x] = tmpmem[0];
}

__global__ void parallel_sum_reduction_consecutive(int *a, int *b, int block_size) {
	int start = blockIdx.x * block_size;
	int step = block_size / 2;

	while (step >= 1) {
		if (threadIdx.x < step) {
			a[start + threadIdx.x] += a[start + threadIdx.x + step];
		}
		__syncthreads();
		step /= 2;
	}
	if (threadIdx.x == 0) b[blockIdx.x] = a[start];
}

__global__ void parallel_sum_reduction_consecutive_with_shared_mem(int *a, int *b, int block_size) {
	__shared__ int tmpmem[THREADS_PER_BLOCK];
	int step = block_size / 2;

	tmpmem[threadIdx.x] = a[threadIdx.x + blockDim.x * blockIdx.x];
	__syncthreads();

	while (step >= 1) {
		if (threadIdx.x < step) {
			tmpmem[threadIdx.x] += tmpmem[threadIdx.x + step];
		}
		__syncthreads();
		step /= 2;
	}
	if (threadIdx.x == 0) b[blockIdx.x] = tmpmem[0];
}

int main() {
	cudaSetDevice(0);
	// set environment
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
	parallel_sum_naive<<<NUM_BLOCKS/MULT, THREADS_PER_BLOCK>>>(dev_arr, dev_output, MULT);
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
	if (gpuSum != expectedSum) printf("GPU error!\n");
	gpuTotalTime -= (gpuSumTime + gpuSetupTime);

	// set environment for parallel reduction
	free(host_output);
	host_output = (int *) malloc(NUM_BLOCKS * sizeof(int));
	cudaFree(dev_output);
	cudaMalloc((void **) &dev_output,  NUM_BLOCKS * sizeof(int));

	// gpu time to calculate sum using parallel reduction
	gstart();
	parallel_sum_reduction<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(dev_arr, dev_output, THREADS_PER_BLOCK);
	gend(&gpuSumTime);
	gpuTotalTime += gpuSumTime;

	gstart();
	gpuSum = 0;
	cudaMemcpy(host_output, dev_output, NUM_BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < NUM_BLOCKS; i++) {
		gpuSum += host_output[i];
	}
	gend(&gpuSetupTime);
	gpuTotalTime += gpuSetupTime;
	printf("gpu sum time (with reduction) = %f, gpu total time = %f\n", gpuSumTime, gpuTotalTime);
	if (gpuSum != expectedSum) printf("GPU error!\n");
	gpuTotalTime -= (gpuSumTime + gpuSetupTime);

	// gpu time to calculate sum using parallel reduction with shared memory
	cudaMemcpy(dev_arr, host_arr, N * sizeof(int), cudaMemcpyHostToDevice);
	gstart();
	parallel_sum_reduction_with_shared_mem<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(dev_arr, dev_output, THREADS_PER_BLOCK);
	gend(&gpuSumTime);
	gpuTotalTime += gpuSumTime;

	gstart();
	gpuSum = 0;
	cudaMemcpy(host_output, dev_output, NUM_BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < NUM_BLOCKS; i++) {
		gpuSum += host_output[i];
	}
	gend(&gpuSetupTime);
	gpuTotalTime += gpuSetupTime;
	printf("gpu sum time (with reduction and shared mem) = %f, gpu total time = %f\n", gpuSumTime, gpuTotalTime);
	if (gpuSum != expectedSum) printf("GPU error! Sum is %lld\n", gpuSum);
	gpuTotalTime -= (gpuSumTime + gpuSetupTime);

	// gpu time to calculate sum using parallel consecutive reduction
	gstart();
	parallel_sum_reduction_consecutive<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(dev_arr, dev_output, THREADS_PER_BLOCK);
	gend(&gpuSumTime);
	gpuTotalTime += gpuSumTime;

	gstart();
	gpuSum = 0;
	cudaMemcpy(host_output, dev_output, NUM_BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < NUM_BLOCKS; i++) {
		gpuSum += host_output[i];
	}
	gend(&gpuSetupTime);
	gpuTotalTime += gpuSetupTime;
	printf("gpu sum time (with consecutive reduction) = %f, gpu total time = %f\n", gpuSumTime, gpuTotalTime);
	if (gpuSum != expectedSum) printf("GPU error!\n");
	gpuTotalTime -= (gpuSumTime + gpuSetupTime);

	// gpu time to calculate sum using parallel consecutive reduction with shared memory
	cudaMemcpy(dev_arr, host_arr, N * sizeof(int), cudaMemcpyHostToDevice);	
	gstart();
	parallel_sum_reduction_consecutive_with_shared_mem<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(dev_arr, dev_output, THREADS_PER_BLOCK);
	gend(&gpuSumTime);
	gpuTotalTime += gpuSumTime;

	gstart();
	gpuSum = 0;
	cudaMemcpy(host_output, dev_output, NUM_BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < NUM_BLOCKS; i++) {
		gpuSum += host_output[i];
	}
	gend(&gpuSetupTime);
	gpuTotalTime += gpuSetupTime;
	printf("gpu sum time (with consecutive reduction and shared mem) = %f, gpu total time = %f\n", gpuSumTime, gpuTotalTime);
	if (gpuSum != expectedSum) printf("GPU error!\n");
	gpuTotalTime -= (gpuSumTime + gpuSetupTime);

	printf("Expected Sum is %lld, cpu Sum is %lld, gpu Sum is %lld\n", expectedSum, cpuSum, gpuSum);
	free(host_arr);
	free(host_output);
	cudaFree(dev_arr);
	cudaFree(dev_output);
}