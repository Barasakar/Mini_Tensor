#pragma once
#include <iostream>
#include "cuda_runtime.h"

#define blocksPerGrid 1
template <typename T> class MiniTensor {
	// initialize a MiniTensor object
public:
	MiniTensor();
	~MiniTensor();

};

template <typename T> MiniTensor<T>::MiniTensor() {
	int gpu_count;
	cudaGetDeviceCount(&gpu_count);
	printf("Current GPU count: %d\n", gpu_count);

	for (int current_gpu = 0; current_gpu < gpu_count; current_gpu++) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, current_gpu);
		printf("Current device %d %s has compute capability %d.%d.\n", current_gpu, deviceProp.name, deviceProp.major, deviceProp.minor);
		printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
		printf("Wrap size: %d\n", deviceProp.warpSize);
		printf("Max Threads Dimension: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf("Register per Thread: %d\n", deviceProp.regsPerBlock / 1024);
		printf("Max Grid Size (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("Shared Memory Per Block: %d bytes\n", deviceProp.sharedMemPerBlock);
	}
}

template <typename T> MiniTensor<T>::~MiniTensor() {
}