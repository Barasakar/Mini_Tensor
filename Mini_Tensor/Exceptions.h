//Exceptions.h
#pragma once
#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

class CudaException : public std::runtime_error {
public:
	CudaException(const std::string& message, cudaError_t error) : std::runtime_error(message), error_(error) {}
	cudaError_t getError() const { return error_; }	
private:
	cudaError_t error_;
};

#endif