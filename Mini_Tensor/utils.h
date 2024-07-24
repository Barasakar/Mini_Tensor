//utils.h
#pragma once
#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cuda_runtime.h>
#include "Exceptions.h"

#define cudaCheckError(call) \
	do { \
		cudaError error = call; \
		if (error != cudaSuccess) { \
			std::string msg = "CUDA Error: " + std::string(cudaGetErrorString(error)) + \
			" in file " +  __FILE__ + " at line " + std::to_string(__LINE__) ; \
			throw CudaException(msg, error); \
		} \
	} while(0) 
#endif

