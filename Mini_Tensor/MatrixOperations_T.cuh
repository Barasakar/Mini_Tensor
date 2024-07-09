// MatrixOperations_T.cuh

#ifndef MINI_TENSOR_MATRIXOPERATIONS_CUH
#define MINI_TENSOR_MATRIXOPERATIONS_CUH

#include "utils.h"
#include "CudaMemoryManagement_T.h"

namespace MiniTensor {
	template<typename T> __global__ void kernel_matrixAdd(T* d_input_1, T* d_input_2, T* d_output, long long total_size) {
		int thread_id = blockId.x * blockDim.x + threadId.x;
		if (thread_id < total_size) {
			d_output[thread_id] = d_input_1[thread_id] + d_input_2[thread_id];
		}
	}

	/// <summary>
	/// The current design for MatrixOperations_T is that it reiceves input and output as pointers to host memory 
	/// and perform operations in the device memory.
	/// </summary>
	/// <typeparam name="T"> Any data type. </typeparam>
	template <typename T> class MatrixOperations {
	public:
		MatrixOperations(long long rows, long long cols);
		~MatrixOperations() = default;

		void add(T* input_1, T* input_2, T* output);

	private:
		long long rows;
		long long cols;
		long long total_size;
		CudaMemoryManagement<T> d_input_1;
		CudaMemoryManagement<T> d_input_2;
		CudaMemoryManagement<T> d_output;
	};


	template <typename T>
	MatrixOperations<T>::MatrixOperations(long long rows, long long cols) : rows(rows), cols(cols), total_size(rows * cols), 
	d_input_1(total_size), d_input_2(total_size), d_output(total_size){}

	/// <summary>
	/// Perform matrix addition given two inputs.
	/// </summary>
	/// <param name="input_1"> input 1 from host memory. </param>
	/// <param name="input_2"> input 2 from host memory. </param>
	/// <param name="output">  output from host memory. </param>
	template <typename T>
	void MatrixOperations<T>::add(T* input_1, T* input_2, T* output) {
		try {
			cudaCheckError(cudaMemcpy(d_input_1.getPtr(), input_1, total_size * sizeof(T), cudaMemcpyHostToDevice));
			cudaCheckError(cudaMemcpy(d_input_2.getPtr(), input_2, total_size * sizeof(T), cudaMemcpyHostToDevice));

			// TODO: need to finetune the block size and grid size based on the user's graphic card.
			kernel_matrixAdd<T> << <(total_size + 255) / 256, 256 >> > (d_input_1.getPtr(), d_input_2.getPtr(), d_output.getPtr(), total_size);

			cudaCheckError(cudaPeekAtLastError());
			cudaCheckError(cudaDeviceSynchronize());


			cudaCheckError(cudaMemcpy(output, d_output.getPtr(), total_size * sizeof(T), cudaMemcpyDeviceToHost));
		}
		catch (const CudaException& e) {
			std::cerr << "MatrixOperations::add failed: " << e.what() << std::endl;

		}
	

	}

}
#endif 
