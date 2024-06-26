#include "MiniTensor_T.h"
#include "MatrixOperations_T.cuh"

int main() {
	MiniTensor<int> *mt = new MiniTensor<int>();
	MatrixOperations<int>* mo = new MatrixOperations<int>(100, 100);
	delete mt;
	delete mo;
	return 0;
}