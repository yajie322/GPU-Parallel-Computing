/*
	This code copys numbers from host to deive and back, prints them.
	It also examines the differences between host pointers and device pointers.
*/


#include <stdio.h>
#include <stdlib.h>
#include "timerc.h"

#define gerror(ans) {gpuAssert((ans), __FILE__, __LINE__);}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ int a[1]; // in GPU global memory
__global__ void printfaddress(int *r) {
	printf("Content of address %p from GPU perspective = %d\n", r, r[0]);
}

__global__ void increase_memory(int *r, int n) {
	int ix = threadIdx.x + blockDim.x * blockIdx.x;

	if (ix < n) {
		r[ix] = r[ix] + ix;
	}
}

int main(void) {
	cudaSetDevice(0);

	int *dev_ptr;
	int n = 128;

	//Set up numbers in CPU memory
	int *host_ptr = (int *) malloc(n*sizeof(int));
	for (int i = 0; i < n; i++) {
		host_ptr[i] = i;
	}

	//Set up memory in GPU and copy numbers to GPU
	cudaMalloc((void**) &dev_ptr, n*sizeof(int));
	cudaMemcpy(dev_ptr, host_ptr, n*sizeof(int), cudaMemcpyHostToDevice);

	dim3 numthreadsperblock(1024,1,1);
	dim3 numblockspergrid((n+1023)/1024, 1, 1);

	increase_memory<<<numblockspergrid, numthreadsperblock>>>(dev_ptr, n);
	cudaMemcpy(host_ptr, dev_ptr, n*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < n; i++) {
		printf("%d ", host_ptr[i]);
	}
	printf("\n");
	free(host_ptr);
	cudaFree(dev_ptr);



	// To show host and device pointers are different things
	int a_host = 134;

	printf("Address of a from CPUs persepctive = %p\n", a);

	int *address_of_a;
	cudaGetSymbolAddress((void**) &address_of_a, a);
	printf("Address of a from GPUs perspective = %p\n", address_of_a);

	cudaMemcpyToSymbol(a, &a_host, sizeof(int), 0, cudaMemcpyHostToDevice);

	printfaddress<<<1,1>>>(address_of_a);
	cudaDeviceSynchronize();

	gerror(cudaPeekAtLastError());
	cudaDeviceSynchronize();





}