#ifndef HEADERS_H
#define HEADERS_H
#include"eigen-3.3.8\Eigen\Dense"
#include<iostream>
#include<vector>
#include<set>
#include<cstdint>
typedef Eigen::MatrixXf emat; // "Eigen" matrix
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> dmat; // matrix of data
typedef Eigen::VectorXf evec; // "Eigen" vector
typedef int64_t int64;
typedef uint64_t uint64;
using namespace std;
#endif