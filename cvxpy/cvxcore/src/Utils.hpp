//   Copyright 2017 Steven Diamond
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

// Some useful defines for Matricies/etc.

#ifndef UTILS_H
#define UTILS_H

#include "../include/Eigen/Core"
#include "../include/Eigen/Sparse"

#define NULL_MATRIX Eigen::SparseMatrix<double>(0, 0)

typedef Eigen::Matrix<int, Eigen::Dynamic, 1> Vector;
typedef Eigen::SparseMatrix<double> Matrix;
typedef std::map<int, Matrix> CoeffMap;
typedef Eigen::Triplet<double> Triplet;
typedef std::map<int, std::map<int, std::vector<Matrix> > > Tensor;
typedef std::map<int, std::vector<Matrix> > DictMat;

/* ID for all things of CONSTANT_TYPE */
static const int CONSTANT_ID = -1;

int vecprod(const std::vector<int> &vec);
int vecprod_before(const std::vector<int> &vec, int end);
Tensor tensor_mul(const Tensor &lh_ten, const Tensor &rh_ten);
void acc_tensor(Tensor &lh_ten, const Tensor &rh_ten);
Matrix diagonalize(const Matrix &mat);

#endif
