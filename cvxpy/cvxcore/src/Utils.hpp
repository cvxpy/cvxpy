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

#include <memory>

#define NULL_MATRIX Eigen::SparseMatrix<double>(0, 0)

typedef Eigen::Matrix<int, Eigen::Dynamic, 1> Vector;
typedef Eigen::SparseMatrix<double> Matrix;

using MatrixFn = std::function<Matrix(const Matrix &)>;

class AbstractLinOp {
/**
 * rows x cols linear operator
 */
public:
  AbstractLinOp(int rows, int cols,
      const MatrixFn &matmul, const MatrixFn &rmatmul)
      : rows_(rows), cols_(cols), matmul_(matmul), rmatmul_(rmatmul), has_matrix_(false) {};
  AbstractLinOp operator+(const AbstractLinOp &obj) const;
  AbstractLinOp operator-(const AbstractLinOp &obj) const;
  AbstractLinOp operator*(const AbstractLinOp &obj) const;

  AbstractLinOp transpose() const {
    AbstractLinOp op = AbstractLinOp(cols_, rows_, rmatmul_, matmul_);
    if (has_matrix()) {
      op.set_matrix(std::make_shared<const Matrix>(get_matrix().transpose()));
    }
    return op;
  }

  int rows() const { return rows_; }
  int cols() const { return cols_; }

  Matrix matmul(const Matrix &matrix) const { return matmul_(matrix); }
  Matrix rmatmul(const Matrix &matrix) const { return rmatmul_(matrix); }

  bool has_matrix() const { return has_matrix_; }
  void set_matrix(std::shared_ptr<const Matrix> matrix) {
    matrix_ = matrix;
    has_matrix_ = true;
  }
  const Matrix& get_matrix() const { return *matrix_; }
  std::shared_ptr<const Matrix> get_matrix_ptr() const { return matrix_; }

private:
  int rows_;
  int cols_;
  MatrixFn matmul_;
  MatrixFn rmatmul_;

  bool has_matrix_;
  std::shared_ptr<const Matrix> matrix_;
};

AbstractLinOp from_matrix(std::shared_ptr<const Matrix> matrix);

typedef Eigen::Triplet<double> Triplet;
typedef std::map<int, std::vector<AbstractLinOp> > DictMat;
typedef std::map<int, std::map<int, std::vector<AbstractLinOp> > > Tensor;

/* ID for all things of CONSTANT_TYPE */
static const int CONSTANT_ID = -1;

int vecprod(const std::vector<int> &vec);
int vecprod_before(const std::vector<int> &vec, int end);
Tensor tensor_mul(const Tensor &lh_ten, const Tensor &rh_ten);
void acc_tensor(Tensor &lh_ten, const Tensor &rh_ten);

Matrix diagonalize(const Matrix &mat);


#endif
