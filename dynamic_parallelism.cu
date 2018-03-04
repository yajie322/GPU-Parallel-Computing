/*
	This code does the parallel sum by using the concept of dynamic parallelism
*/

#include <stdio.h>
#include <stdlib.h>
#include "timerc.h"

#define K 10
#define N (1 << K)
#define THREADS_PER_BLOCK 32
#define NUM_BLOCKS N/2/THREADS_PER_BLOCK

__global__ void naive_recursive_sum(int *a, int n, int start) {
	if (n == 1) {
		if (threadIdx.x == 0) {
			a[start] += a[start+n];
		}
	}
	else {
		naive_recursive_sum<<<1,2>>>(a, n/2, start + threadIdx.x * n);
		__syncthreads();
		if (threadIdx.x == 0) {
			cudaDeviceSynchronize();
			a[start] += a[start+n];
		}
	}
}

__global__ void better_recursive_sum(int *a, int *b) {
	int size = 2 * blockDim.x;
	int *a_fixed = a + blockIdx.x * size;
	int *b_fixed = b + blockIdx.x;

	if (size == 2 && threadIdx.x == 0) {
		b_fixed[0] = a_fixed[0] + a_fixed[1];
	} 
	else {
		size /= 2;

		if(size > 1 && threadIdx.x < size) {
        	a_fixed[threadIdx.x] += a_fixed[threadIdx.x + size];
    	}
    	__syncthreads();
    
    	if(threadIdx.x == 0){
    		better_recursive_sum<<<1,size/2>>>(a_fixed, b_fixed);
    	}
    }
}


int main() {
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, K+1);
	int *host_arr = (int *) malloc(N * sizeof(int));
	int *dev_arr;
	cudaMalloc((void **) &dev_arr, N * sizeof(int));
	for (int i = 0; i < N; i++) {
		host_arr[i] = 1;
	}
	cudaMemcpy(dev_arr, host_arr, N * sizeof(int), cudaMemcpyHostToDevice);
	naive_recursive_sum<<<1,2>>>(dev_arr, N/2, 0);
	int result;
	cudaMemcpy(&result, dev_arr, sizeof(int), cudaMemcpyDeviceToHost);
	printf("error (naive) is %d\n", result - N);

	int* host_output = (int *) malloc(NUM_BLOCKS * sizeof(int));
	int *dev_output;
	cudaMalloc((void **) &dev_output, NUM_BLOCKS * sizeof(int));
	cudaMemcpy(dev_arr, host_arr, N * sizeof(int), cudaMemcpyHostToDevice);
	better_recursive_sum<<<NUM_BLOCKS,THREADS_PER_BLOCK>>>(dev_arr, dev_output);
	cudaMemcpy(host_output, dev_output, NUM_BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);
	result = 0;
	for (int i = 0; i < NUM_BLOCKS; i++) {
		result += host_output[i];
	}
	printf("error (better) is %d\n", result - N);

	return 0;
}