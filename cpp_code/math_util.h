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


// find differences between every pair
// of vector entries that are 
// next to each other
evec derivatives(const evec& vec) {
    evec dx(vec.size() - 1);
    dx.setZero();

    for (int i = 1; i < vec.size(); i++) {
        int im1 = i - 1;
        dx[im1] = vec[i] - vec[im1];
    }

    return dx;
}


// multiply the matrix by itself a certain number of times,
// uses eigenvalue decomposition
emat pow(const emat& mat, double p) {
	Eigen::EigenSolver<emat> eigsolver(mat);

	emat eigvals = eigsolver.pseudoEigenvalueMatrix();
	emat eigmat = eigsolver.pseudoEigenvectors();

	int64_t size = min(eigvals.rows(), eigvals.cols());

	for (int64_t i = 0; i < size; i++) {
		eigvals(i, i) = pow(eigvals(i, i), p);
	}

	return eigmat * eigvals * eigmat.inverse();
}


// turn the matrix into a markov matrix, 
// so the columns represent probabilities
void markovify(emat& mat) {
	for (int64_t i = 0; i < mat.cols(); i++) {
		mat.col(i) /= (double)mat.col(i).sum();
	}
}


// raise the matrix coefficients to a power
void cwisePow(emat& mat, double p) {
	for (int64_t i = 0; i < mat.rows(); i++) {
		for (int64_t j = 0; j < mat.cols(); j++) {
			mat(i, j) = pow(mat(i, j), p);
		}
	}
}


// go through the rows of the matrix 
// and normalize each of them
void normalizeRows(emat& mat) {
	for (int64_t i = 0; i < mat.rows(); i++) {
		mat.row(i).normalize();
	}
}


// go through the rows of the matrix 
// and normalize each of them
void normalizeRows(dmat& mat) {
	for (int64_t i = 0; i < mat.rows(); i++) {
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
vector<int64_t> rankByRule(const evec& vec, bool descending=true) {
	vector<int64_t> perm = permutationVector<int64_t>(vec.size());

	if (descending) {
		sort(perm.begin(), perm.end(), [&vec](int64_t lhs, int64_t rhs)
			{ return vec[lhs] > vec[rhs]; });
	}
	else {
		sort(perm.begin(), perm.end(), [&vec](int64_t lhs, int64_t rhs)
			{ return vec[lhs] < vec[rhs]; });
	}

	return perm;
}


// get the index of the highest vector entry
int64_t indexOfHighest(const evec& vec) {
	int64_t highestInd = 0;
	float highestVal = vec[0];

	for (int64_t i = 1; i < vec.size(); i++) {
		if (vec[i] > highestVal) {
			highestInd = i;
			highestVal = vec[i];
		}
	}

	return highestInd;
}


// get the index of the lowest vector entry
int64_t indexOfLowest(const evec& vec) {
	int64_t lowestInd = 0;
	float lowestVal = vec[0];

	for (int64_t i = 1; i < vec.size(); i++) {
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
    float ave = 0.0f;
    for (size_t i = 0; i < size; i++) {
        ave += arr[i];
    }
    ave /= (double)size;
    
    for (size_t i = 0; i < size; i++) {
        arr[i] = sigmoid(arr[i], ave, 16);
    }
}


// calculate the entropy of the vector
// using the entropy equation. 
// values closer to 0.5 are highest, and
// values closer to 1 or 0 are lowest
float calcEntropy(const evec& x) {
	int64_t size = x.size();
	float entropy = 0.0f;
	for (int64_t i = 0; i < size; i++) {
		float p = x[i];
		float q = 1.0 - p;
		entropy += (-p * log2(p)) - (q * log2(q));
	}
	return entropy / (float)size;
}

#endif