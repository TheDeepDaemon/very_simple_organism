#ifndef DATA_AUGMENTATION_H
#define DATA_AUGMENTATION_H
#include"cpp_headers.h"



void rotate90CounterClockwise(const dmat& data, Eigen::Map<dmat>& rotated) {
	for (int64_t i = 0; i < data.rows(); i++) {
		for (int64_t j = 0; j < data.cols(); j++) {
			rotated(rotated.rows() - j - 1, i) = data(i, j);
		}
	}
}


void rotate90Clockwise(const dmat& data, Eigen::Map<dmat>& rotated) {
	for (int64_t i = 0; i < data.rows(); i++) {
		for (int64_t j = 0; j < data.cols(); j++) {
			rotated(j, rotated.cols() - i - 1) = data(i, j);
		}
	}
}


void rotate180(const dmat& data, Eigen::Map<dmat>& rotated) {
	for (int64_t i = 0; i < data.rows(); i++) {
		for (int64_t j = 0; j < data.cols(); j++) {
			rotated(rotated.rows() - i - 1, rotated.cols() - j - 1) = data(i, j);
		}
	}
}


void fliph(const dmat& data, Eigen::Map<dmat>& flipped) {
	for (int64_t i = 0; i < data.rows(); i++) {
		for (int64_t j = 0; j < data.cols(); j++) {
			flipped(i, data.cols() - j - 1) = data(i, j);
		}
	}
}


void flipv(const dmat& data, Eigen::Map<dmat>& flipped) {
	for (int64_t i = 0; i < data.rows(); i++) {
		for (int64_t j = 0; j < data.cols(); j++) {
			flipped(data.rows() - i - 1, j) = data(i, j);
		}
	}
}

template<bool square>
void augmentData(float* dataPtr, 
	const int64_t n, const int64_t rows, const int64_t cols, 
	float* augmentedDataPtr) {

	// if the 90 degree rotations are taken out,
	// the new size isn't going to be as big
	const int64_t augmentedSize = square ? 6 : 4;

	// store the full size of each section
	int64_t matSize = rows * cols;

	for (int64_t i = 0; i < n; i++) {
		// get original data
		Eigen::Map<dmat> mat(dataPtr + (i * matSize), rows, cols);

		float* outPtr = augmentedDataPtr + (i * matSize * augmentedSize);

		// pass on original values
		Eigen::Map<dmat> out0(outPtr, rows, cols);
		out0 = mat;

		// flip horizontally, add to new data
		Eigen::Map<dmat> out1(outPtr + (matSize * 1), rows, cols);
		fliph(mat, out1);

		// flip vertically, add to new data
		Eigen::Map<dmat> out2(outPtr + (matSize * 2), rows, cols);
		flipv(mat, out2);
		
		// rotate the matrix 180 degrees, add to new data
		Eigen::Map<dmat> out3(outPtr + (matSize * 3), rows, cols);
		rotate180(mat, out3);

		// if 90 degree rotations are turned on
		if (square) {
			// rotate the matrix 90 degrees, add to new data
			Eigen::Map<dmat> out4(outPtr + (matSize * 4), cols, rows);
			rotate90CounterClockwise(mat, out4);

			// rotate the matrix 90 degrees, add to new data
			Eigen::Map<dmat> out5(outPtr + (matSize * 5), cols, rows);
			rotate90Clockwise(mat, out5);
		}
		
	}
}

#endif