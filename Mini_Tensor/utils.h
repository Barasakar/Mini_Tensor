#pragma once
#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cuda_runtime.h>

#define cudaCheckError(call) \
	do { \
		cudaError error = call; \
		if (error != cudaSuccess) { \
			std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; \
			exit(error); \
		} \
	} while(0) 
#endif