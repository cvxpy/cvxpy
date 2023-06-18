//   Copyright 2023, the CVXPY developers
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#ifndef SPARSE_CHOLESKY_H
#define SPARSE_CHOLESKY_H

#include "../../cvxcore/include/Eigen/Core"
#include "../../cvxcore/include/Eigen/Sparse"
#include "../../cvxcore/include/Eigen/SparseCholesky"
#include <vector>
#include <string>

#define NULL_MATRIX Eigen::SparseMatrix<double>(0, 0)

typedef Eigen::Matrix<int, Eigen::Dynamic, 1> Vector;
typedef Eigen::SparseMatrix<double> Matrix;
typedef std::map<int, Matrix> CoeffMap;
typedef Eigen::Triplet<double> Triplet;
typedef std::map<int, std::map<int, std::vector<Matrix> > > Tensor;
typedef std::map<int, std::vector<Matrix> > DictMat;



struct CholeskyFailure : public std::runtime_error {
  CholeskyFailure(const std::string& msg) : std::runtime_error{msg} {}
};

Matrix sparse_from_vecs(
    int n_rows,
    int n_cols,
    std::vector<int> &rows,
    std::vector<int> &cols,
    std::vector<double> vals
);

void vecs_from_sparse(
    Matrix mat,
    std::vector<int> &rows,
    std::vector<int> &cols,
    std::vector<double> &vals
);

void sparse_chol_from_vecs(
    int n,
    std::vector<int> &in_rows,
    std::vector<int> &in_cols,
    std::vector<double> &in_vals,
    std::vector<int> &out_pivots,
    std::vector<int> &out_rows,
    std::vector<int> &out_cols,
    std::vector<double> &out_vals
);


#endif