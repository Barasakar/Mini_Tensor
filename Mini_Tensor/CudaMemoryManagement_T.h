#pragma once
#ifndef CUDA_MEMORY_MANAGEMENT_T_H
#define CUDA_MEMORY_MANAGEMENT_T_H
#include "utils.h"


template <typename T>
class CudaMemoryManagement {
public:
	CudaMemoryManagement(size_t size) : size_(size), ptr(nullptr) {
		cudaCheckError(cudaMalloc(&ptr_, size_ * sizeof(T)));
	}
	~CudaMemoryManagement() {
		if (ptr_) {
			cudaCheckError(cudaFree(ptr_));
		}
	}
	/// <summary>
	/// Get the pointer from the current object.
	/// </summary>
	/// <returns> A pointer </returns>
	T* getPtr() const {
		return ptr_;
	}

	//Disable both copy constructor and assignment operator to avoid double free errors as copy and assignment
	// operators might lead to multiple objects managing the same resource.
	CudaMemoryManagement(const CudaMemoryManagement&) = delete;
	CudaMemoryManagement& operator=(const CudaMemoryManagement&) = delete;

	

private:
	size_t size_;
	T* ptr_;
};

#endif