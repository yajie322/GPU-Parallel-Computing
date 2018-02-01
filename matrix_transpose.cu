/*
	This piece of codes returns a tranposed matrix
*/

#include <stdio.h>
#include "timerc.h"

#define N 8192
#define DIM 32

__global__ void matrix_transpose_xfirst(int *imat, int *omat, int n) {
	/*
		this function transposes matrix by read x first (because as threadIdx.x increments by 1, 
		s increments by 1 as well), which is more efficient that yfirst due to cache.
	*/
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int s = iy * n + ix;
	int d = ix * n + iy;
	omat[s] = imat[d];
}

__global__ void matrix_transpose_yfirst(int *imat, int *omat, int n) {
	/*
		this function transposes matrix by read y first (because as threadIdx.y increments by 1, 
		s increments by 1 as well), which is less efficient
	*/
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int s = ix * n + iy;
	int d = iy * n + ix;
	omat[s] = imat[d];
}

__global__ void matrix_transpose_shared_memory(int *imat, int *omat, int n) {
	/*
		this function transposes matrix using shared memory. This method is much more efficient 
		because we both read and write x first.
	*/
	__shared__ int tmpmat[DIM * DIM];
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int s = iy * n + ix;
	int ix2 = threadIdx.x + blockDim.x * blockIdx.y;
	int iy2 = threadIdx.y + blockDim.y * blockIdx.x;
	int d = iy2 * n + ix2;
	tmpmat[threadIdx.x + DIM * threadIdx.y] = imat[s];

	__syncthreads();

	omat[d] = tmpmat[threadIdx.y + threadIdx.x * DIM];
}

__global__ void matrix_transpose_without_bank_conflicts(int *imat, int *omat, int n) {
	/*
		this function transposes matrix using shared memory but without bank conflicts. 
		This method is  more efficient because now threads in the same warp access memory using 
		different banks.
	*/
	__shared__ int tmpmat[DIM * (DIM + 1)];
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int s = iy * n + ix;
	int ix2 = threadIdx.x + blockDim.x * blockIdx.y;
	int iy2 = threadIdx.y + blockDim.y * blockIdx.x;
	int d = iy2 * n + ix2;
	tmpmat[threadIdx.x + (DIM + 1) * threadIdx.y] = imat[s];

	__syncthreads();

	omat[d] = tmpmat[threadIdx.y + threadIdx.x * (DIM + 1)];
}


int main() {
	cudaSetDevice(0);
	int *dev_imat;
	int *dev_omat;

	float time;

	int *imat = (int *) malloc(N * N * sizeof(int));
	int *omat = (int *) malloc(N * N * sizeof(int));

	// Test CPU time
	for (int i = 0; i < N * N; i++) imat[i] = i;
	cstart();
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			omat[i * N + j] = imat[j * N + i];
		}
	}
	cend(&time);
	printf("transpose cpu time = %f\n", time);

	// Set up for GPU test
	cudaMalloc((void**) &dev_imat, N * N * sizeof(int));
	cudaMalloc((void**) &dev_omat, N * N * sizeof(int));
	cudaMemcpy(dev_imat, imat, N * N * sizeof(int), cudaMemcpyHostToDevice);

	// Test GPU times for different methods
	dim3 num_of_blocks(N/DIM, N/DIM);
	dim3 threads_per_block(DIM, DIM);
	gstart();
	matrix_transpose_xfirst<<<num_of_blocks,threads_per_block>>>(dev_imat, dev_omat, N);
	gend(&time);
	printf("transpose xfirst gpu time = %f\n", time);
	gstart();
	matrix_transpose_yfirst<<<num_of_blocks,threads_per_block>>>(dev_imat, dev_omat, N);
	gend(&time);
	printf("transpose yfirst gpu time = %f\n", time);
	gstart();
	matrix_transpose_shared_memory<<<num_of_blocks,threads_per_block>>>(dev_imat, dev_omat, N);
	gend(&time);
	printf("transpose with shared memory gpu time = %f\n", time);
	gstart();
	matrix_transpose_without_bank_conflicts<<<num_of_blocks,threads_per_block>>>(dev_imat, dev_omat, N);
	gend(&time);
	printf("transpose with shared memory and no bank conflicts gpu time = %f\n", time);

	cudaMemcpy(omat, dev_omat, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	// for (int i = 0; i < N; i++) {
	// 	for (int j = 0; j < N; j++) {
	// 		printf("%d ", omat[i * N + j]);
	// 	}
	// 	printf("\n");
	// }

	free(omat);
	free(imat);
	cudaFree(dev_imat);
	cudaFree(dev_omat);
	return 0;
}
