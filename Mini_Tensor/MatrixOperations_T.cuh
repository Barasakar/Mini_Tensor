// MatrixOperations_T.cuh
// The current design for MatrixOperations_T is that it performs on the device memory.
#ifndef MINI_TENSOR_MATRIXOPERATIONS_CUH
#define MINI_TENSOR_MATRIXOPERATIONS_CUH

#include <cuda_runtime.h>


template<typename T = int> __global__ void kernel_matrixAdd(T* input_1, T* input_2, T* output, long long total_size) {
	int thread_id = blockId.x * blockDim.x + threadIdx.x;
	if (thread_id < total_size) {
		output[thread_id] = input_1[thread_id] + input_2[thread_id];
	}
}

template <typename T = int> class MatrixOperations {
public:
	MatrixOperations();
	MatrixOperations(long long rows, long long cols);
	~MatrixOperations();

	void add(T* input_1, T* input_2, T* output);
private:
	long long rows;
	long long cols;
	long long total_size;
	T* d_input_1;
	T* d_input_2;
	T* d_output;
};

template <typename T> 
MatrixOperations<T>::MatrixOperations() {
	rows = 0;
	cols = 0;
	total_size = 0;
	d_input_1 = nullptr;
	d_input_2 = nullptr;
	d_output = nullptr;
}

template <typename T>
MatrixOperations<T>::MatrixOperations(long long rows, long long cols) {
	this->rows = rows;
	this->cols = cols;
	total_size = rows * cols;

	// Allocate memory on device
	cudaMalloc((void**)&d_input_1, total_size * sizeof(T));
	cudaMalloc((void**)&d_input_2, total_size * sizeof(T));
	cudaMalloc((void**)&d_output, total_size * sizeof(T));

}

template <typename T>
MatrixOperations<T>::~MatrixOperations() {
	cudaFree(d_input_1);
	cudaFree(d_input_2);
	cudaFree(d_output);
}

#endif 
