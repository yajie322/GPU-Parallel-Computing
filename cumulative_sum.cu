/*
	Given a array of integers, this piece of code examines several methods that compute the
	cumulative sums of the array, i.e. in the resuting array, the n-th element is the sum of 1st
	through n-th elements in the original array.
*/

#include <stdio.h>
#include <stdlib.h>
#include "timerc.h"

#define N 1024*128
#define THREADSPERBLOCK 1024

__global__ void cumulative_sum(int *a, int *b) {
	int size = 2 * blockDim.x;
	int start = 2 * blockDim.x * blockIdx.x;

	for (int step = 1; step < size; step *= 2) {
		if (threadIdx.x < blockDim.x / step) {
			a[start + 2 * step - 1 + threadIdx.x * step * 2] +=
				a[start + step - 1 + threadIdx.x * step * 2];
		}
		__syncthreads();
	}

	for (int step = size / 2; step > 1; step /= 2) {
		if (threadIdx.x < (size / step - 1)) {
			a[start + step - 1 + step / 2 + threadIdx.x * step] +=
				a[start + step - 1 + threadIdx.x * step];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		b[blockIdx.x] = a[start + size - 1];
	}
}

__global__ void sum_blocks(int *a) {
	int size = 2 * blockDim.x;
	int start = 2 * blockDim.x * blockIdx.x;

	for (int step = 1; step < size; step *= 2) {
		if (threadIdx.x < blockDim.x / step) {
			a[start + 2 * step - 1 + threadIdx.x * step * 2] +=
				a[start + step - 1 + threadIdx.x * step * 2];
		}
		__syncthreads();
	}

	for (int step = size / 2; step > 1; step /= 2) {
		if (threadIdx.x < (size / step - 1)) {
			a[start + step - 1 + step / 2 + threadIdx.x * step] +=
				a[start + step - 1 + threadIdx.x * step];
		}
		__syncthreads();
	}
}

__global__ void fix_sum(int *a, int *b, int size) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= size) {
		a[id] += b[blockIdx.x/2 - 1];
	}
}

__global__ void cumulative_sum_shared_mem(int *a, int *b) {
	int size = 2 * blockDim.x;
	__shared__ int tmp[2 * THREADSPERBLOCK];

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	tmp[2 * threadIdx.x] = a[2 * id];
	tmp[2 * threadIdx.x + 1] = a[2 * id + 1];
	__syncthreads();

	for (int step = 1; step < size; step *= 2) {
		if (threadIdx.x < blockDim.x / step) {
			tmp[2 * step - 1 + threadIdx.x * step * 2] +=
				tmp[step - 1 + threadIdx.x * step * 2];
		}
		__syncthreads();
	}

	for (int step = size / 2; step > 1; step /= 2) {
		if (threadIdx.x < (size / step - 1)) {
			tmp[step - 1 + step / 2 + threadIdx.x * step] +=
				tmp[step - 1 + threadIdx.x * step];
		}
		__syncthreads();
	}

	a[2 * id] = tmp[2 * threadIdx.x];
	a[2 * id + 1] = tmp[2 * threadIdx.x + 1];
	__syncthreads();

	if (threadIdx.x == 0) {
		b[blockIdx.x] = tmp[size - 1];
	}
}

__global__ void cumulative_sum_2(int *a, int *b) {
	int start = 2 * blockDim.x * blockIdx.x;
	int size = 2 * blockDim.x;

	for (int step = 1; step <= blockDim.x; step *= 2) {
		a[start + step + 2 * step * (threadIdx.x / step) + threadIdx.x % step]
			+= a[start + step + 2 * step * (threadIdx.x / step) - 1];
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		b[blockIdx.x] = a[start + size - 1];
	}
}


int main() {
	float time;
	int *dev_arr, *dev_output;
	int *host_arr = (int *) malloc(N * sizeof(int));
	int *host_output = (int *) malloc(N * sizeof(int));
	
	for (int i = 0; i < N; i++) {
		host_arr[i] = 1;
	}

	cstart();
	host_output[0] = host_arr[0];
	for (int i = 1; i < N; i++) {
		host_output[i] = host_output[i-1] + host_arr[i];
	}
	cend(&time);
	printf("cpu time = %f\n", time);

	cudaMalloc((void **) &dev_arr, N * sizeof(int));
	cudaMalloc((void **) &dev_output, N * sizeof(int));
	cudaMemcpy(dev_arr, host_arr, N * sizeof(int), cudaMemcpyHostToDevice);

	gstart();
	cumulative_sum<<<N/THREADSPERBLOCK/2,THREADSPERBLOCK>>>(dev_arr, dev_output);
	sum_blocks<<<1, 1024>>>(dev_output);
	fix_sum<<<N/THREADSPERBLOCK, THREADSPERBLOCK>>>(dev_arr, dev_output, 2 * THREADSPERBLOCK);
	gend(&time);
	printf("gpu time = %f\n", time);

	cudaMemcpy(host_output, dev_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
	// for (int i = 0; i < N; i++) {
	// 	printf("%d ", host_output[i]);
	// }
	// printf("\n");

	cudaMemcpy(dev_arr, host_arr, N * sizeof(int), cudaMemcpyHostToDevice);
	gstart();
	cumulative_sum_shared_mem<<<N/THREADSPERBLOCK/2, THREADSPERBLOCK>>>(dev_arr, dev_output);
	sum_blocks<<<1, 1024>>>(dev_output);
	fix_sum<<<N/THREADSPERBLOCK, THREADSPERBLOCK>>>(dev_arr, dev_output, 2 * THREADSPERBLOCK);
	gend(&time);
	printf("gpu time (with shared memory)%f\n", time);

	cudaMemcpy(host_output, dev_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
	// for (int i = 0; i < N; i++) {
	// 	printf("%d ", host_output[i]);
	// }
	// printf("\n");

	cudaMemcpy(dev_arr, host_arr, N * sizeof(int), cudaMemcpyHostToDevice);
	gstart();
	cumulative_sum_2<<<N/THREADSPERBLOCK/2, THREADSPERBLOCK>>>(dev_arr, dev_output);
	sum_blocks<<<1, 1024>>>(dev_output);
	fix_sum<<<N/THREADSPERBLOCK, THREADSPERBLOCK>>>(dev_arr, dev_output, 2 * THREADSPERBLOCK);
	gend(&time);
	printf("gpu time (with 2nd scheme)%f\n", time);

	cudaMemcpy(host_output, dev_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
	// for (int i = 0; i < N; i++) {
	// 	printf("%d ", host_output[i]);
	// }
	// printf("\n");


	return 0;
}