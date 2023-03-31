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

#ifndef LINOP_H
#define LINOP_H

#include "Utils.hpp"
#include <cassert>
#include <iostream>
#include <vector>

/* TYPE of each LinOP */
enum operatortype {
  VARIABLE,
  PARAM,
  PROMOTE,
  MUL,
  RMUL,
  MUL_ELEM,
  DIV,
  SUM,
  NEG,
  INDEX,
  TRANSPOSE,
  SUM_ENTRIES,
  TRACE,
  RESHAPE,
  DIAG_VEC,
  DIAG_MAT,
  UPPER_TRI,
  CONV,
  HSTACK,
  VSTACK,
  SCALAR_CONST,
  DENSE_CONST,
  SPARSE_CONST,
  NO_OP,
  KRON, // for backwards compatibility; equivalent to KRON_R (which is preferred)
  KRON_R,
  KRON_L
};

/* linOp TYPE */
typedef operatortype OperatorType;

/* LinOp Class mirrors the CVXPY linOp class. Data fields are determined
         by the TYPE of LinOp. No error checking is performed on the data
   fields,
         and the semantics of SIZE, ARGS, and DATA depends on the linOp TYPE. */
class LinOp {
public:
  LinOp(OperatorType type, const std::vector<int> &shape,
        const std::vector<const LinOp *> &args)
      : type_(type), shape_(shape), args_(args), sparse_(false),
        data_has_been_set_(false) {}

  OperatorType get_type() const { return type_; }
  bool is_constant() const {
    return type_ == SCALAR_CONST || type_ == DENSE_CONST ||
           type_ == SPARSE_CONST;
  }
  std::vector<int> get_shape() const { return shape_; }

  const std::vector<const LinOp *> get_args() const { return args_; }
  const std::vector<std::vector<int> > get_slice() const { return slice_; }
  void push_back_slice_vec(const std::vector<int> &slice_vec) {
    slice_.push_back(slice_vec);
  }

  bool has_numerical_data() const { return data_has_been_set_; }
  const LinOp *get_linOp_data() const { return linOp_data_; }
  void set_linOp_data(const LinOp *tree) {
    assert(!data_has_been_set_);
    linOp_data_ = tree;
    data_has_been_set_ = true;
  }
  int get_data_ndim() const { return data_ndim_; }
  void set_data_ndim(int ndim) { data_ndim_ = ndim; }
  bool is_sparse() const { return sparse_; }
  const Matrix &get_sparse_data() const { return sparse_data_; }
  const Eigen::MatrixXd &get_dense_data() const { return dense_data_; }

  /* Initializes DENSE_DATA. MATRIX is a pointer to the data of a 2D
   * numpy array, ROWS and COLS are the size of the ARRAY.
   *
   * MATRIX must be a contiguous array of doubles aligned in fortran
   * order.
   *
   * NOTE: The function prototype must match the type-map in CVXCanon.i
   * exactly to compile and run properly.
   */
  void set_dense_data(double *matrix, int rows, int cols) {
    assert(!data_has_been_set_);
    dense_data_ = Eigen::Map<Eigen::MatrixXd>(matrix, rows, cols);
    sparse_ = false;
    data_has_been_set_ = true;
  }

  /* Initializes SPARSE_DATA from a sparse matrix in COO format.
   * DATA, ROW_IDXS, COL_IDXS are assumed to be contiguous 1D numpy arrays
   * where (DATA[i], ROW_IDXS[i], COLS_IDXS[i]) is a (V, I, J) triplet in
   * the matrix. ROWS and COLS should refer to the size of the matrix.
   *
   * NOTE: The function prototype must match the type-map in CVXCanon.i
   * exactly to compile and run properly.
   */
  void set_sparse_data(double *data, int data_len, double *row_idxs,
                       int rows_len, double *col_idxs, int cols_len, int rows,
                       int cols) {
    assert(!data_has_been_set_);
    assert(rows_len == data_len && cols_len == data_len);
    sparse_ = true;
    Matrix sparse_coeffs(rows, cols);
    std::vector<Triplet> tripletList;
    tripletList.reserve(data_len);
    for (int idx = 0; idx < data_len; idx++) {
      tripletList.push_back(
          Triplet(int(row_idxs[idx]), int(col_idxs[idx]), data[idx]));
    }
    sparse_coeffs.setFromTriplets(tripletList.begin(), tripletList.end());
    sparse_coeffs.makeCompressed();
    sparse_data_ = sparse_coeffs;
    data_ndim_ = 2;
    data_has_been_set_ = true;
  }

private:
  const OperatorType type_;
  std::vector<int> shape_;
  // children of this LinOp
  std::vector<const LinOp *> args_;
  // stores slice data as (row_slice, col_slice), where slice = (start, end,
  // step_size)
  std::vector<std::vector<int> > slice_;
  // numerical data acted upon by this linOp
  const LinOp *linOp_data_;
  int data_ndim_;
  // true iff linOp has sparse_data; at most one of sparse_data and
  // dense_data should be set
  bool sparse_;
  Matrix sparse_data_;
  Eigen::MatrixXd dense_data_;
  bool data_has_been_set_;
};
#endif
