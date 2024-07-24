#include "pch.h"

namespace MiniTensor {
	class MatrixOperationsTests : public ::testing::Test {
		protected:
			static constexpr long long cols = 10; // why do I need static constexpr here?
			static constexpr long long rows = 10;

			MatrixOperations<int> matrixOp_int;
			MatrixOperations<double> matrixOp_double;

			int input_1_int[rows * cols];
			int input_2_int[rows * cols];
			int output_int[rows * cols];
			int expected_output_int[rows * cols];

			double input_1_double[rows * cols];
			double input_2_double[rows * cols];
			double output_double[rows * cols];
			double expected_output_double[rows * cols];

			MatrixOperationsTests() : matrixOp_int(rows, cols), matrixOp_double(rows, cols) {
				for (int i = 0; i < rows * cols; i++) {
					input_1_int[i] = i;
					input_2_int[i] = i;
					expected_output_int[i] = i + i;

					input_1_double[i] = static_cast<double>(i);
					input_2_double[i] = static_cast<double>(i);
					expected_output_double[i] = static_cast<double>(i) + static_cast<double>(i);
				}
			
			}
	};
	TEST_F(MatrixOperationsTests, Addition_int) {
		matrixOp_int.add(input_1_int, input_2_int, output_int);
		for (int i = 0; i < rows * cols; i++) {
			EXPECT_EQ(output_int[i], expected_output_int[i]);
		}
	}

	TEST_F(MatrixOperationsTests, Addition_double) {
		matrixOp_double.add(input_1_double, input_2_double, output_double);
		for (int i = 0; i < rows * cols; i++) {
			EXPECT_EQ(output_double[i], expected_output_double[i]);
		}
	}

	TEST_F(MatrixOperationsTests, Addition_int_zero) {
		std::fill_n(input_1_int, rows * cols, 0);
		std::fill_n(input_2_int, rows * cols, 0);
		matrixOp_int.add(input_1_int, input_2_int, output_int);
		for (int i = 0; i < rows * cols; i++) {
			EXPECT_EQ(output_int[i], 0);
		}
	}

	TEST_F(MatrixOperationsTests, Addition_double_zero) {
		std::fill_n(input_1_int, rows * cols, 0.0);
		std::fill_n(input_2_int, rows * cols, 0.0);
		matrixOp_int.add(input_1_int, input_2_int, output_int);
		for (int i = 0; i < rows * cols; i++) {
			EXPECT_EQ(output_int[i], 0.0);
		}
	}
	
}


int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}