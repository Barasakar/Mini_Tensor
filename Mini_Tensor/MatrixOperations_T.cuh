#ifndef MINI_TENSOR_MATRIXOPERATIONS_CUH
#define MINI_TENSOR_MATRIXOPERATIONS_CUH

#include "utils.h"
#include "CudaMemoryManagement_T.h"

namespace MiniTensor {

    template<typename T> __global__ void kernel_matrixAdd(T* d_input_1, T* d_input_2, T* d_output, long long total_size) {
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x; 
        if (thread_id < total_size) {
            d_output[thread_id] = d_input_1[thread_id] + d_input_2[thread_id];
        }
    }

    //https://siboehm.com/articles/22/CUDA-MMM
    template<typename T> __global__ void kernel_matrixMult(T* d_input_1, T* d_input_2, T* d_output, 
        long long rows, long long cols, long long shared_dim, long long total_size) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x < cols && y < rows) {
			T tmp_sum = 0.0;
            for (int i = 0; i < shared_dim; i++) {
                tmp_sum += d_input_1[y * shared_dim + i] * d_input_2[i * cols + x];
            }
			d_output[y * cols + x] = tmp_sum;
        }
    }

    /// <summary>
    /// The current design for MatrixOperations_T is that it receives input and output as pointers to host memory 
    /// and perform operations in the device memory.
    /// </summary>
    /// <typeparam name="T"> Any data type. </typeparam>
    template <typename T>
    class MatrixOperations {
    public:
        MatrixOperations(long long rows, long long cols);
        ~MatrixOperations() = default;

        void add(T* input_1, T* input_2, T* output);
		void multiply(T* input_1, T* input_2, T* output, long long cols, long long rows, long long shared_dim);

    private:
        long long rows;
        long long cols;
        long long total_size;
        CudaMemoryManagement<T> d_input_1;
        CudaMemoryManagement<T> d_input_2;
        CudaMemoryManagement<T> d_output;
    };

    template <typename T>
    MatrixOperations<T>::MatrixOperations(long long rows, long long cols)
        : rows(rows), cols(cols), total_size(rows* cols),
        d_input_1(total_size), d_input_2(total_size), d_output(total_size) {}

    template <typename T>
    void MatrixOperations<T>::add(T* input_1, T* input_2, T* output) {
        try {
            cudaCheckError(cudaMemcpy(d_input_1.getPtr(), input_1, total_size * sizeof(T), cudaMemcpyHostToDevice));
            cudaCheckError(cudaMemcpy(d_input_2.getPtr(), input_2, total_size * sizeof(T), cudaMemcpyHostToDevice));

            kernel_matrixAdd<T> << <(total_size + 255) / 256, 256 >> > (d_input_1.getPtr(), d_input_2.getPtr(), d_output.getPtr(), total_size);

            cudaCheckError(cudaPeekAtLastError());
            cudaCheckError(cudaDeviceSynchronize());

            cudaCheckError(cudaMemcpy(output, d_output.getPtr(), total_size * sizeof(T), cudaMemcpyDeviceToHost));
        }
        catch (const CudaException& e) {
            std::cerr << "MatrixOperations::add failed: " << e.what() << std::endl;
            throw;
        }
    }

	template <typename T>
	void MatrixOperations<T>::multiply(T* input_1, T* input_2, T* output, long long cols, long long rows, long long shared_dim) {
        try {
			cudaCheckError(cudaMemcpy(d_input_1.getPtr(), input_1, total_size * sizeof(T), cudaMemcpyHostToDevice));
			cudaCheckError(cudaMemcpy(d_input_2.getPtr(), input_2, total_size * sizeof(T), cudaMemcpyHostToDevice));

			kernel_matrixMult<T> << <(total_size + 255) / 256, 256 >> > (d_input_1.getPtr(), d_input_2.getPtr(), d_output.getPtr(), rows, cols, shared_dim, total_size);

			cudaCheckError(cudaPeekAtLastError());
			cudaCheckError(cudaDeviceSynchronize());

			cudaCheckError(cudaMemcpy(output, d_output.getPtr(), total_size * sizeof(T), cudaMemcpyDeviceToHost));
		}
        catch (const CudaException& e) {
			std::cerr << "MatrixOperations::multiply failed: " << e.what() << std::endl;
			throw;
		}
	}

} // namespace MiniTensor

#endif // MINI_TENSOR_MATRIXOPERATIONS_CUH
