/*
	This code tests times taken to perform differnt tasks on GPU vs CPU.
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


__global__ void test_warp_divergence1() {
	if (threadIdx.x < 16) {
		double v = 0;
		for (int i = 0; i < 1000; i++) {
			for (int j = 0; j < i * i; j++) {
				v = v + i + i * j / 3.435;
			}
		}
	} else {
		double v = 0;
		for (int i = 0; i < 1000; i++) {
			for (int j = 0; j < i * i; j++) {
				v = v + i - i * j / 1.435;
			}
		}
	}
}

__global__ void test_warp_divergence2() {
	if (threadIdx.x < 8) {
		double v = 0;
		for (int i = 0; i < 1000; i++) {
			for (int j = 0; j < i * i; j++) {
				v = v + i + i * j / 3.435;
			}
		}
	} else if (threadIdx.x < 16) {
		double v = 0;
		for (int i = 0; i < 1000; i++) {
			for (int j = 0; j < i * i; j++) {
				v = v + i + i * j / 3.435;
			}
		}
	} else if (threadIdx.x < 24) {
		double v = 0;
		for (int i = 0; i < 1000; i++) {
			for (int j = 0; j < i * i; j++) {
				v = v + i + i * j / 3.435;
			}
		}
	} else if (threadIdx.x < 32) {
		double v = 0;
		for (int i = 0; i < 1000; i++) {
			for (int j = 0; j < i * i; j++) {
				v = v + i + i * j / 3.435;
			}
		}
	} 
}

__global__ void gpucycle() {
	double v = 0;
	for (int i = 0; i < 1000; i++) {
		for (int j = 0; j < i * i; j++) {
			v = v + i + i * j / 3.435;
		}
	}
}


void cpucycle() {
	double v = 0;
	for (int i = 0; i < 1000; i++) {
		for (int j = 0; j < i * i; j++) {
			v = v + i + i * j / 3.435;
		}
	}
}

int main(void) {
	cudaSetDevice(0);

	float time = 0;
	cstart();
	cpucycle();
	cend(&time);
	printf("cpu time = %f\n", time);
	fflush(stdout);

	gstart();
	gpucycle<<<1,32>>>();
	gend(&time);
	printf("gpu time = %f\n", time);
	fflush(stdout);

	gstart();
	test_warp_divergence1<<<1,32>>>();
	gend(&time);
	printf("gpu time = %f\n", time);
	fflush(stdout);

	gstart();
	test_warp_divergence2<<<1,32>>>();
	gend(&time);
	printf("gpu time = %f\n", time);
	fflush(stdout);


	return 0;
}


