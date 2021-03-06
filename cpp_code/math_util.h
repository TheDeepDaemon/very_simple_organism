#ifndef MATH_UTIL_H
#define MATH_UTIL_H
#include"cpp_headers.h"


inline float sigmoid(float x, float offset, float slope) {
    return 1.0f / (1.0f + exp(-slope * (x - offset)));
}


// returns the intersection of two sets
template<typename T>
set<T> intersectSets(const set<T>& set1, const set<T>& set2) {
	set<T> intersection;
	set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
		inserter(intersection, intersection.begin()));
	return intersection;
}


// returns the union of two sets
template<typename T>
set<T> unifySets(const set<T>& set1, const set<T>& set2) {
	set<T> setUnion;
	set_union(set1.begin(), set1.end(), set2.begin(), set2.end(),
		inserter(setUnion, setUnion.begin()));
	return setUnion;
}


// find dx / di
// or changes in x
evec derivatives(const evec& vec) {
    evec dx(vec.size() - 1);
    dx.setZero();

    for (int i = 1; i < vec.size(); i++) {
        int iLast = i - 1;
        dx[iLast] = vec[i] - vec[iLast];
    }

    return dx;
}


// change in proportion to the previous
evec percentChanges(const evec& vec) {
    evec dx(vec.size() - 1);
    dx.setZero();

    for (int i = 1; i < vec.size(); i++) {
        int iLast = i - 1;
        dx[iLast] = (vec[i] - vec[iLast]) / vec[iLast];
    }

    return dx;
}





// multiply the matrix by itself a certain number of times,
// uses eigenvalue decomposition
emat pow(const emat& mat, double p) {
	Eigen::EigenSolver<emat> eigsolver(mat);

	emat eigvals = eigsolver.pseudoEigenvalueMatrix();
	emat eigmat = eigsolver.pseudoEigenvectors();

	int64 size = min(eigvals.rows(), eigvals.cols());

	for (int64 i = 0; i < size; i++) {
		eigvals(i, i) = pow(eigvals(i, i), p);
	}

	return eigmat * eigvals * eigmat.inverse();
}


// turn the matrix into a markov matrix, 
// so the columns represent probabilities
void markovify(emat& mat) {
	for (int64 i = 0; i < mat.cols(); i++) {
		mat.col(i) /= (double)mat.col(i).sum();
	}
}


// raise the matrix coefficients to a power
void cwisePow(emat& mat, double p) {
	for (int64 i = 0; i < mat.rows(); i++) {
		for (int64 j = 0; j < mat.cols(); j++) {
			mat(i, j) = pow(mat(i, j), p);
		}
	}
}


// go through the rows of the matrix 
// and normalize each of them
void normalizeRows(emat& mat) {
	for (int64 i = 0; i < mat.rows(); i++) {
		mat.row(i).normalize();
	}
}


// go through the rows of the matrix 
// and normalize each of them
void normalizeRows(dmat& mat) {
	for (int64 i = 0; i < mat.rows(); i++) {
		mat.row(i).normalize();
	}
}


// get a vector with integer entries that correspond to indices
template<typename T>
vector<T> permutationVector(size_t size) {
	vector<T> perm(size, 0);
	for (size_t i = 0; i < size; i++) {
		perm[i] = T(i);
	}
	return perm;
}


// return a permutation vector that 
// goes from highest to lowest or vice versa
vector<int64> rankByRule(const evec& vec, bool descending=true) {
	vector<int64> perm = permutationVector<int64>(vec.size());

	if (descending) {
		sort(perm.begin(), perm.end(), [&vec](int64 lhs, int64 rhs)
			{ return vec[lhs] > vec[rhs]; });
	}
	else {
		sort(perm.begin(), perm.end(), [&vec](int64 lhs, int64 rhs)
			{ return vec[lhs] < vec[rhs]; });
	}

	return perm;
}


// get the index of the highest vector entry
int64 indexOfHighest(const evec& vec) {
	int64 highestInd = 0;
	float highestVal = vec[0];

	for (int64 i = 1; i < vec.size(); i++) {
		if (vec[i] > highestVal) {
			highestInd = i;
			highestVal = vec[i];
		}
	}

	return highestInd;
}


// get the index of the lowest vector entry
int64 indexOfLowest(const evec& vec) {
	int64 lowestInd = 0;
	float lowestVal = vec[0];

	for (int64 i = 1; i < vec.size(); i++) {
		if (vec[i] < lowestVal) {
			lowestInd = i;
			lowestVal = vec[i];
		}
	}

	return lowestInd;
}


// applies a sigmoid function to all entries in the array
// the sigmoid is centered on the average value, 
// and it has a slope of 16
void applyAdjustedSigmoid(float* arr, size_t size) {
	// find the average value
    float ave = 0.0f;
    for (size_t i = 0; i < size; i++) {
        ave += arr[i];
    }
    ave /= (double)size;
    
	// apply the sigmoid function
    for (size_t i = 0; i < size; i++) {
        arr[i] = sigmoid(arr[i], ave, 16);
    }
}


// calculate the entropy of the vector
// using the entropy equation. 
// values closer to 0.5 are highest, and
// values closer to 1 or 0 are lowest
float calcEntropy(const evec& x) {
	int64 size = x.size();
	float entropy = 0.0f;
	for (int64 i = 0; i < size; i++) {
		float p = x[i];
		float q = 1.0 - p;
		entropy += (-p * log2(p)) - (q * log2(q));
	}
	return entropy / (float)size;
}


// get the values that the neural network
// weights should be set to based on the data
void toWeightValues(float* weightsPtr, int64 size) {
	Eigen::Map<evec> weights(weightsPtr, size);
	float average = weights.sum() / (float)size;
	weights.array() -= average;
	weights *= (2.0f / weights.cwiseAbs().sum());
}


inline float& sampleMatrix(
	float* dataPtr, const int64 i, const int64 j, 
	const int64 k, const int64 cols, const int64 classes) {
	return dataPtr[(i * cols * classes) + (j * classes) + k];
}


#endif