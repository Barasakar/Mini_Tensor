// MatrixOperations_T.cuh

#ifndef MINI_TENSOR_MATRIXOPERATIONS_CUH
#define MINI_TENSOR_MATRIXOPERATIONS_CUH

#include "utils.h"

template<typename T> __global__ void kernel_matrixAdd(T* input_1, T* input_2, T* output, long long total_size) {
	int thread_id = blockId.x * blockDim.x + threadIdx.x;
	if (thread_id < total_size) {
		output[thread_id] = input_1[thread_id] + input_2[thread_id];
	}
}

/// <summary>
/// The current design for MatrixOperations_T is that it reiceves input and output as pointers to host memory 
/// and perform operations in the device memory.
/// </summary>
/// <typeparam name="T"> Any data type. </typeparam>
template <typename T> class MatrixOperations {
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
	cudaCheckError(cudaMalloc((void**)&d_input_1, total_size * sizeof(T)));
	cudaCheckError(cudaMalloc((void**)&d_input_2, total_size * sizeof(T)));
	cudaCheckError(cudaMalloc((void**)&d_output, total_size * sizeof(T)));

}

template <typename T>
MatrixOperations<T>::~MatrixOperations() {
	cudaCheckError(cudaFree(d_input_1));
	cudaCheckError(cudaFree(d_input_2));
	cudaCheckError(cudaFree(d_output));
}

/// <summary>
/// Perform matrix addition given two inputs.
/// </summary>
/// <param name="input_1"> input 1 from host memory. </param>
/// <param name="input_2"> input 2 from host memory. </param>
/// <param name="output">  output from host memory. </param>
template <typename T>
void MatrixOperations<T>::add(T* input_1, T* input_2, T* output) {

}

#endif 
