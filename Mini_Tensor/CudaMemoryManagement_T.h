#pragma once
#ifndef CUDA_MEMORY_MANAGEMENT_T_H
#define CUDA_MEMORY_MANAGEMENT_T_H
#include "utils.h"


template <typename T>
class CudaMemoryManagement {
public:
	CudaMemoryManagement(size_t size) : size_(size), ptr_(nullptr) {
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

	//Move constructor and assignment operator
	CudaMemoryManagement(CudaMemoryManagement&& other) noexcept : size_(other.size_), ptr_(other.ptr_) {
		other.ptr_ = nullptr;
		other.size_ = 0;
	}
	CudaMemoryManagement& operator=(CudaMemoryManagement&& other) noexcept {
		if (this != &other) {
			if (ptr_) { //used ptr_ instead of other.ptr_ because we want to check if this.ptr_ has any ownership or not.
				cudaCheckError(cudaFree(ptr_));
			}
			// Update and nullify:
			ptr_ = other.ptr_;
			size_ = other.size_;
			other.ptr_ = nullptr;
			other.size_ = 0;
		}
		return *this;
	}

private:
	size_t size_;
	T* ptr_;
};

#endif