#ifndef CLUSTERING_H
#define CLUSTERING_H
#include"cpp_headers.h"
#include"math_util.h"




// returns a matrix representing the similarity 
// of each row to the others. the inner product
// is used to judge similarity
emat dataRelationships(emat dataMatrix) {
	normalizeRows(dataMatrix);
	dataMatrix = dataMatrix * dataMatrix.transpose();
	//dataMatrix.diagonal().setZero();
	markovify(dataMatrix);
	return dataMatrix;
}



// checks to see if the columns have 
// converged to 0 or 1 values (or close)
bool allColsConverged(const emat& mat) {
	for (int64 i = 0; i < mat.cols(); i++) {

		// count non-zero values
		int count = 0;
		for (int64 j = 0; j < mat.rows(); j++) {
			double d = mat(j, i);
			if (d != 0.0) {
				count++;
			}
		}

		double desired = 1.0 / (double)count;

		for (int64 j = 0; j < mat.rows(); j++) {
			double mc = mat(j, i);
			if (mc != 0.0 && abs(mc - desired) > 0.00001) {
				return false;
			}
		}

	}
	return true;
}


// an algorithm that multiplies a markov matrix by itself
// and then raises the coefficients to a power, until they have converged
// this is good for finding clusters in a graph
emat markovClustering(emat mat, uint64_t maxIterations, double power, double inflation) {

	emat lastMat = mat; // matrix at last iteration

	for (uint64_t i = 0; i < maxIterations; i++) {
		// power
		mat = pow(mat, power);

		// inflation
		cwisePow(mat, inflation);
		
		markovify(mat);

		// end if done
		if (allColsConverged(mat)) {
			return mat;
		}

		// end early if the process starts to
		// give corrupted results
		if (!mat.hasNaN() && mat.allFinite()) {
			lastMat = mat;
		}
		else {
			return lastMat;
		}
	}

	return mat;
}


// find groupings in the data, which is given as a matrix
vector<set<int64>> groupData(const emat& data, uint64_t iterations, double power, double inflation) {
	
	emat clusteredData = markovClustering(dataRelationships(data), iterations, power, inflation);

	for (int64 i = 0; i < clusteredData.rows(); i++) {
		for (int64 j = 0; j < clusteredData.cols(); j++) {
			if (abs(clusteredData(i, j)) < 0.0001) {
				clusteredData(i, j) = 0.0;
			}
		}
	}

	vector<set<int64>> groups(clusteredData.cols());

	for (int64 i = 0; i < clusteredData.cols(); i++) {
		double maxColVal = 0.0;

		// find max value that isn't on the diagonal,
		// meaning it is not a self-relation
		for (int64 j = 0; j < clusteredData.rows(); j++) {
			if (i != j) {
				if (clusteredData(j, i) > maxColVal) {
					maxColVal = clusteredData(j, i);
				}
			}
		}

		groups[i].insert(i);
		if (maxColVal > 0.0) {
			// find highest value
			for (int64 j = 0; j < clusteredData.rows(); j++) {
				if (i != j) {
					if (clusteredData(j, i) >= maxColVal) {
						groups[i].insert(j);
					}
				}
			}
		}
	}

	vector<bool> merged(groups.size(), false);
	
	// keep going until there are no more things to merge
	bool mergedSomething = true;
	while (mergedSomething) {
		mergedSomething = false;
		for (size_t i = 0; i < groups.size(); i++) {
			for (size_t j = 0; j < groups.size(); j++) {
				if (i != j && (merged[i] == false && merged[j] == false)) {

					set<int64> intersection = intersectSets(groups[i], groups[j]);

					if (!intersection.empty()) {

						groups[i] = unifySets(groups[i], groups[j]);

						merged[j] = true;
						mergedSomething = true;
					}
				}
			}
		}
	}

	vector<set<int64>> finalGroups;

	for (size_t i = 0; i < groups.size(); i++) {
		if (!merged[i]) {
			finalGroups.push_back(groups[i]);
		}
	}

	return finalGroups;
}

#endif