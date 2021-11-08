#ifndef GROUPING_H
#define GROUPING_H
#include"cpp_headers.h"
#include"math_util.h"


// get the differences between different rows,
// treating rows as vectors
dmat getRowDifferences(const dmat& mat) {
	int64 rows = mat.rows();
	dmat diffs(rows, rows);
	
	diffs.setZero();
	
	for (int64 i = 0; i < rows; i++) {
		for (int64 j = 0; j < rows; j++) {
			if (i != j) {
				// squared norm gives better results than regular norm
				diffs(i, j) = (mat.row(i) - mat.row(j)).squaredNorm();
			}
		}
	}
	
	return diffs;
}


// get the relationships between different rows,
// using normalized dot product as a measure of similarity
dmat getDpRelations(dmat mat) {
	normalizeRows(mat);
	dmat relations = mat * mat.transpose();
	relations.diagonal().setZero();
	return relations;
}


// get the sum of each row
evec sumRows(dmat m) {
	evec v(m.rows());
	for (int64 i = 0; i < m.rows(); i++) {
		v[i] = m.row(i).sum();
	}
	return v;
}


// how similar is each row to the rest of the data?
evec getRelations(const dmat& data) {
	evec dotProdRelations = sumRows(getDpRelations(data));
	
	evec vectorDiffRelations = sumRows(getRowDifferences(data));
	
	// gets scores that rank how similar each vector is to the rest,
	// takes distance and dot product into account.
	evec relations = dotProdRelations.cwiseQuotient(vectorDiffRelations);

	return relations;
}


// get the index of any row that is a member of the cluster.
// it assumes that most of the rows in the matrix show a similar pattern.
// similarity is detirmined using dot product and difference.
vector<int64> getIndicesInGroup(const dmat& data) {
	if (data.rows() > 2) {
		evec relations = getRelations(data);
		
		// get the indices
		vector<int64> rankings = rankByRule(relations);
		evec ranked(rankings.size());

		// set values in order from greatest to least
		for (size_t i = 0; i < rankings.size(); i++) {
			ranked[i] = relations[rankings[i]];
		}

		// find the greatest dropoff, looking for lowest second delta
		int64 inflectionPoint = indexOfLowest(derivatives(derivatives(ranked)));
		
		int64 endIndex = inflectionPoint + 2;
		
		// get the outlier index vector
		vector<int64> inGroup(rankings.begin(), rankings.begin() + endIndex);
		return inGroup;
	}
	else {
		return permutationVector<int64>(data.rows());
	}
}


// get the row that serves the best example of the pattern that
// the cluster shows, and compare all other rows to it
vector<int64> compareToHighest(const dmat& data, const evec& relations) {

	int64 highest = indexOfHighest(relations);

	evec dpSimilarity(relations.size());
	for (int64 i = 0; i < relations.size(); i++) {
		dpSimilarity[i] = data.row(i).normalized().dot(data.row(highest).normalized());
	}

	evec distSimilarity(relations.size());
	for (int64 i = 0; i < relations.size(); i++) {
		distSimilarity[i] = (data.row(i) - data.row(highest)).norm();
	}

	evec similarity = dpSimilarity.cwiseQuotient(distSimilarity);

	similarity[highest] = 0;
	
	vector<int64> rankings = rankByRule(similarity);

	size_t numToKeep = 3;

	vector<int64> topChoices;
	topChoices.reserve(numToKeep);
	topChoices.push_back(highest);
	for (size_t i = 0; i < numToKeep-1; i++) {
		topChoices.push_back(rankings[i]);
	}

	return topChoices;
}

#endif