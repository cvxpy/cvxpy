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

#include "LinOpOperations.hpp"
#include "LinOp.hpp"
#include "Utils.hpp"
#include <cassert>
#include <iostream>
#include <map>

/***********************
 * FUNCTION PROTOTYPES *
 ***********************/
Tensor build_tensor(const Matrix &mat);
Tensor get_sum_coefficients(const LinOp &lin, int arg_idx);
Tensor get_sum_entries_mat(const LinOp &lin, int arg_idx);
Tensor get_trace_mat(const LinOp &lin, int arg_idx);
Tensor get_neg_mat(const LinOp &lin, int arg_idx);
Tensor get_div_mat(const LinOp &lin, int arg_idx);
Tensor get_promote_mat(const LinOp &lin, int arg_idx);
Tensor get_mul_mat(const LinOp &lin, int arg_idx);
Tensor get_mul_elemwise_mat(const LinOp &lin, int arg_idx);
Tensor get_rmul_mat(const LinOp &lin, int arg_idx);
Tensor get_index_mat(const LinOp &lin, int arg_idx);
Tensor get_transpose_mat(const LinOp &lin, int arg_idx);
Tensor get_reshape_mat(const LinOp &lin, int arg_idx);
Tensor get_diag_vec_mat(const LinOp &lin, int arg_idx);
Tensor get_diag_matrix_mat(const LinOp &lin, int arg_idx);
Tensor get_upper_tri_mat(const LinOp &lin, int arg_idx);
Tensor get_conv_mat(const LinOp &lin, int arg_idx);
Tensor get_hstack_mat(const LinOp &lin, int arg_idx);
Tensor get_vstack_mat(const LinOp &lin, int arg_idx);
Tensor get_kronr_mat(const LinOp &lin, int arg_idx);
Tensor get_kronl_mat(const LinOp &lin, int arg_idx);
Tensor get_variable_coeffs(const LinOp &lin, int arg_idx);
Tensor get_const_coeffs(const LinOp &lin, int arg_idx);
Tensor get_param_coeffs(const LinOp &lin, int arg_idx);

// TODO: Many functions in this file make unnecessary copies of data;
//       these copies should be eliminated.

/**
 * Computes a vector of coefficient matrices for the linOp LIN based on the
 * type of linOp.
 *
 * Note: This function assumes LIN has been initialized with the correct
 * data, size, and arguments for each linOp type. No error-checking or
 * error-handling for these types of errors is performed.
 *
 * Parameters: LinOp node LIN
 *
 * Returns: std::vector of sparse coefficient matrices for LIN
 */
Tensor get_node_coeffs(const LinOp &lin, int arg_idx) {
  Tensor coeffs;
  switch (lin.get_type()) {
  case VARIABLE:
    coeffs = get_variable_coeffs(lin, arg_idx);
    break;
  case SCALAR_CONST:
    coeffs = get_const_coeffs(lin, arg_idx);
    break;
  case DENSE_CONST:
    coeffs = get_const_coeffs(lin, arg_idx);
    break;
  case SPARSE_CONST:
    coeffs = get_const_coeffs(lin, arg_idx);
    break;
  case PARAM:
    coeffs = get_param_coeffs(lin, arg_idx);
    break;
  case PROMOTE:
    coeffs = get_promote_mat(lin, arg_idx);
    break;
  case MUL:
    coeffs = get_mul_mat(lin, arg_idx);
    break;
  case RMUL:
    coeffs = get_rmul_mat(lin, arg_idx);
    break;
  case MUL_ELEM:
    coeffs = get_mul_elemwise_mat(lin, arg_idx);
    break;
  case DIV:
    coeffs = get_div_mat(lin, arg_idx);
    break;
  case SUM:
    coeffs = get_sum_coefficients(lin, arg_idx);
    break;
  case NEG:
    coeffs = get_neg_mat(lin, arg_idx);
    break;
  case INDEX:
    coeffs = get_index_mat(lin, arg_idx);
    break;
  case TRANSPOSE:
    coeffs = get_transpose_mat(lin, arg_idx);
    break;
  case SUM_ENTRIES:
    coeffs = get_sum_entries_mat(lin, arg_idx);
    break;
  case TRACE:
    coeffs = get_trace_mat(lin, arg_idx);
    break;
  case RESHAPE:
    coeffs = get_reshape_mat(lin, arg_idx);
    break;
  case DIAG_VEC:
    coeffs = get_diag_vec_mat(lin, arg_idx);
    break;
  case DIAG_MAT:
    coeffs = get_diag_matrix_mat(lin, arg_idx);
    break;
  case UPPER_TRI:
    coeffs = get_upper_tri_mat(lin, arg_idx);
    break;
  case CONV:
    coeffs = get_conv_mat(lin, arg_idx);
    break;
  case HSTACK:
    coeffs = get_hstack_mat(lin, arg_idx);
    break;
  case VSTACK:
    coeffs = get_vstack_mat(lin, arg_idx);
    break;
  case KRON_R:
    coeffs = get_kronr_mat(lin, arg_idx);
    break;
  case KRON_L:
    coeffs = get_kronl_mat(lin, arg_idx);
    break;
  case KRON:
    // here for backwards compatibility
    coeffs = get_kronr_mat(lin, arg_idx);
    break;
  default:
    std::cerr << "Error: linOp type invalid." << std::endl;
    exit(-1);
  }
  return coeffs;
}

Tensor lin_to_tensor(const LinOp &lin) {
  if (lin.get_args().size() == 0) {
    return get_node_coeffs(lin, 0);
  } else {
    Tensor result;
    /* Multiply the arguments of the function coefficient in order */
    for (unsigned i = 0; i < lin.get_args().size(); ++i) {
      Tensor lh_coeff = get_node_coeffs(lin, i);
      Tensor rh_coeff = lin_to_tensor(*lin.get_args()[i]);
      Tensor prod = tensor_mul(lh_coeff, rh_coeff);
      acc_tensor(result, prod);
    }
    return result;
  }
}

/*******************
 * HELPER FUNCTIONS
 *******************/

/**
 * Returns a vector containing the sparse matrix mat
 *
 * NB: This function takes ownership of mat!
 */
Tensor build_tensor(Matrix &mat) {
  Tensor ten;
  ten[CONSTANT_ID] = DictMat();
  DictMat* dm = &(ten[CONSTANT_ID]);
  (*dm)[CONSTANT_ID] = std::vector<Matrix>();

  std::vector<Matrix>* mat_vec = &(*dm)[CONSTANT_ID];
  mat_vec->push_back(Matrix());
  // swap the contents of &mat with the newly constructed matrix,
  // instead of copying it into the vector.
  (*mat_vec)[0].swap(mat);
  return ten;
}

/**
 * Returns an N x N sparse identity matrix.
 */
Matrix sparse_eye(int n) {
  Matrix eye_n(n, n);
  eye_n.setIdentity();
  return eye_n;
}

/**
 * Returns a sparse ROWS x COLS matrix of all ones.
 *
 * TODO: This function returns a sparse representation of a dense matrix,
 * which might not be extremely efficient, but does make it easier downstream.
 */
Matrix sparse_ones(int rows, int cols) {
  Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(rows, cols);
  return ones.sparseView();
}

// Returns a sparse rows x cols matrix with matrix[row_sel, col_sel] = 1.
Matrix sparse_selector(int rows, int cols, int row_sel, int col_sel) {
  Matrix selector(rows * cols, 1);
  selector.insert(row_sel + rows * col_sel, 0) = 1.0;
  return selector;
}

/**
 * Reshapes the input matrix into a single column vector that preserves
 * columnwise ordering. Equivalent to Matlab's (:) operation.
 *
 * Params: sparse Eigen matrix MAT of size ROWS by COLS.
 * Returns: sparse Eigen matrix OUT of size ROWS * COLS by 1
 */

Matrix sparse_reshape_to_vec(const Matrix &mat) {
  int rows = mat.rows();
  int cols = mat.cols();
  Matrix out(rows * cols, 1);
  std::vector<Triplet> tripletList;
  tripletList.reserve(rows * cols);
  for (int k = 0; k < mat.outerSize(); ++k) {
    for (Matrix::InnerIterator it(mat, k); it; ++it) {
      tripletList.push_back(Triplet(it.col() * rows + it.row(), 0, it.value()));
    }
  }
  out.setFromTriplets(tripletList.begin(), tripletList.end());
  return out;
}

/******************
 * The remaining helper functions are all used to retrieve constant
 * data from the linOp object. Depending on the interface to the calling
 * package, the implementation of these functions will have to change
 * accordingly!
 ******************/

/**
 * Returns the matrix stored in the data field of LIN as a sparse eigen matrix
 * If COLUMN is true, the matrix is reshaped into a column vector which
 * preserves the columnwise ordering of the elements, equivalent to
 * matlab (:) operator.
 *
 * Note all matrices are returned in a sparse representation to force
 * sparse matrix operations in build_matrix.
 *
 * Params: LinOp LIN with DATA containing a 2d vector representation of a
 * 				 matrix. boolean COLUMN
 *
 * Returns: sparse eigen matrix COEFFS
 *
 * TODO: this function unnecessarily copies data out of lin; instead of
 *       returning a Matrix, this function should take a pointer to a Matrix,
 *       and it should make the Matrix point to the data in lin (without
 *       copying it)
 *
 */
Matrix get_constant_data(const LinOp &lin, bool column) {
  assert(lin.has_numerical_data());
  Matrix coeffs;
  if (lin.is_sparse()) {
    if (column) {
      coeffs = sparse_reshape_to_vec(lin.get_sparse_data());
    } else {
      coeffs = lin.get_sparse_data();
    }
  } else {
    assert(lin.get_dense_data().rows() > 0);
    assert(lin.get_dense_data().cols() > 0);
    if (column) {
      Eigen::Map<const Eigen::MatrixXd> column(
          lin.get_dense_data().data(),
          lin.get_dense_data().rows() * lin.get_dense_data().cols(), 1);
      coeffs = column.sparseView();
    } else {
      coeffs = lin.get_dense_data().sparseView();
    }
  }
  coeffs.makeCompressed();
  return coeffs;
}

/**
 * Interface for the VARIABLE linOp to retrieve its variable ID.
 *
 * Parameters: linOp LIN of type VARIABLE with a variable ID in the
 * 							0,0 component of the
 * DENSE_DATA matrix.
 *
 * Returns: integer variable ID
 */
int get_id_data(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == VARIABLE || lin.get_type() == PARAM);
  return int(lin.get_dense_data()(0, 0));
}

/*****************************
 * LinOP -> Matrix FUNCTIONS
 *****************************/
/**
 * Return the coefficients for KRON_R.
 *
 * Parameters: linOp LIN with type KRON_R
 * Returns: vector containing the coefficient matrix for the Kronecker
            product with a Variable in the right operand
 */
Tensor get_kronr_mat(const LinOp &lin, int arg_idx) {
  // This function doesn't properly canonicalize LinOp objects derived from CVXPY Parameters.
  // See get_mul_elemwise_mat (or other multiplication functions other than kronr)
  // for examples of correct parameter handling.
  assert(lin.get_type() == KRON_R);
  Matrix lh = get_constant_data(*lin.get_linOp_data(), false);
  int lh_rows = lh.rows();
  int rh_rows = lin.get_args()[0]->get_shape()[0];
  int rh_cols = lin.get_args()[0]->get_shape()[1];
  int rh_size = rh_rows * rh_cols;

  Matrix mat(lh_rows * lh.cols() * rh_size, rh_size);

  std::vector<Triplet> tripletList;
  tripletList.reserve(rh_size * lh.nonZeros());
  int kron_rows = lh_rows * rh_rows;
  int row_offset, col;
  double val;
  for (int k = 0; k < lh.outerSize(); ++k) {
    for (Matrix::InnerIterator it(lh, k); it; ++it) {
      row_offset = (kron_rows * rh_cols) * it.col() + rh_rows * it.row();
      val = it.value();
      col = 0;
      for (int j = 0; j < rh_cols; ++j) {
        for (int i = 0; i < rh_rows; ++i) {
          tripletList.push_back(Triplet(row_offset + i, col, val));
          col++;
        }
        row_offset += kron_rows;  // hit this rh_cols many times.
      }
    }
  }
  mat.setFromTriplets(tripletList.begin(), tripletList.end());
  mat.makeCompressed();
  return build_tensor(mat);
}

/**
 * Return the coefficients for KRON_L.
 *
 * Parameters: linOp LIN with type KRON_L
 * Returns: vector containing the coefficient matrix for the Kronecker
            product with a Variable in left operand
 */
Tensor get_kronl_mat(const LinOp &lin, int arg_idx) {
  // This function doesn't properly canonicalize LinOp objects derived from CVXPY Parameters.
  // See get_mul_elemwise_mat (or other multiplication functions other than kronr)
  // for examples of correct parameter handling.
  assert(lin.get_type() == KRON_L);
  Matrix rh = get_constant_data(*lin.get_linOp_data(), false);
  int rh_rows = rh.rows();
  int rh_cols = rh.cols();
  int lh_rows = lin.get_args()[0]->get_shape()[0];
  int lh_cols = lin.get_args()[0]->get_shape()[1];

  // Construct row indices for the first column of mat.
  //   We rely on the fact that rh is an Eigen sparse matrix,
  //   and assume its storage order is CSC. Note that Eigen's
  //   default order is CSC-like, and when "compressed" the storage
  //   is actually CSC.
  assert(!rh.IsRowMajor);
  int row_offset = 0;
  int kron_rows = lh_rows * rh_rows;
  int rh_nnz = rh.nonZeros();
  std::vector<int> base_row_indices;
  std::vector<double> vec_rh;
  base_row_indices.reserve(rh_nnz);
  vec_rh.reserve(rh_nnz);
  for (int k = 0; k < rh.outerSize(); ++k) {  // loop over columns
  	for (Matrix::InnerIterator it(rh, k); it; ++it) { // loop over nonzeros in this column
  	  int cur_row = it.row() + row_offset;
  	  base_row_indices.push_back(cur_row);
  	  vec_rh.push_back(it.value());
  	}
  	row_offset += kron_rows;
  }

  int lh_size = lh_rows * lh_cols;
  int rh_size = rh_rows * rh_cols;
  Matrix mat(lh_size * rh_size, lh_size);
  std::vector<Triplet> tripletList;
  tripletList.reserve(lh_size * rh_nnz);

  int row, col;
  int outer_row_offset = 0;
  for (int j = 0; j < lh_cols; ++j) {
  	row_offset = outer_row_offset;
  	for (int i = 0; i < lh_rows; ++i) {
  	  col = i + j * lh_rows;
  	  for (int ell = 0; ell < rh_nnz; ++ell) {
  	  	row = base_row_indices[ell] + row_offset;
  	  	tripletList.push_back(Triplet(row , col, vec_rh[ell]));
  	  }
  	  row_offset += rh_rows;
  	}
  	outer_row_offset += lh_rows * rh_size;
  }
  mat.setFromTriplets(tripletList.begin(), tripletList.end());
  mat.makeCompressed();
  return build_tensor(mat);
}

/**
 * Return the coefficients for VSTACK.
 *
 * Parameters: linOp LIN with type VSTACK
 * Returns: vector of coefficient matrices for each argument.
 */
Tensor get_vstack_mat(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == VSTACK);
  int row_offset = 0;
  assert(static_cast<size_t>(arg_idx) <= lin.get_args().size());
  std::vector<Triplet> tripletList;
  const LinOp &arg = *lin.get_args()[arg_idx];
  tripletList.reserve(vecprod(arg.get_shape()));

  int arg_rows = (arg.get_shape().size() >= 2) ? arg.get_shape()[0] : 1;
  int arg_cols = (arg.get_shape().size() >= 1)
                     ? arg.get_shape()[arg.get_shape().size() - 1]
                     : 1;
  /* Columns are interleaved. */
  int column_offset = lin.get_shape()[0];
  for (int idx = 0; idx < arg_idx; ++idx) {
    const LinOp &prev_arg = *lin.get_args()[idx];
    row_offset +=
        (prev_arg.get_shape().size() >= 2) ? prev_arg.get_shape()[0] : 1;
  }

  for (int i = 0; i < arg_rows; ++i) {
    for (int j = 0; j < arg_cols; ++j) {
      int row_idx = i + (j * column_offset) + row_offset;
      int col_idx = i + (j * arg_rows);
      tripletList.push_back(Triplet(row_idx, col_idx, 1));
    }
  }

  Matrix coeff(vecprod(lin.get_shape()), vecprod(arg.get_shape()));
  coeff.setFromTriplets(tripletList.begin(), tripletList.end());
  coeff.makeCompressed();
  return build_tensor(coeff);
}

/**
 * Return the coefficients for HSTACK
 *
 * Parameters: linOp LIN with type HSTACK
 * Returns: vector of coefficient matrices for each argument.
 */
Tensor get_hstack_mat(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == HSTACK);
  int row_offset = 0;
  assert(static_cast<size_t>(arg_idx) <= lin.get_args().size());
  std::vector<Triplet> tripletList;
  tripletList.reserve(vecprod(lin.get_shape()));
  const LinOp &arg = *lin.get_args()[arg_idx];

  int arg_rows = (arg.get_shape().size() >= 1) ? arg.get_shape()[0] : 1;
  int arg_cols = (arg.get_shape().size() >= 2) ? arg.get_shape()[1] : 1;
  /* Columns are laid out in order. */
  int column_offset = arg_rows;
  for (int idx = 0; idx < arg_idx; ++idx) {
    row_offset += vecprod(lin.get_args()[idx]->get_shape());
  }

  for (int i = 0; i < arg_rows; ++i) {
    for (int j = 0; j < arg_cols; ++j) {
      int row_idx = i + (j * column_offset) + row_offset;
      int col_idx = i + (j * arg_rows);
      tripletList.push_back(Triplet(row_idx, col_idx, 1));
    }
  }

  Matrix coeff(vecprod(lin.get_shape()), vecprod(arg.get_shape()));
  coeff.setFromTriplets(tripletList.begin(), tripletList.end());
  coeff.makeCompressed();
  return build_tensor(coeff);
}

/**
 * Return the coefficients for CONV operator. The coefficient matrix is
 * constructed by creating a toeplitz matrix with the constant vector
 * in DATA as the columns. Multiplication by this matrix is equivalent
 * to convolution.
 *
 * Parameters: linOp LIN with type CONV. Data should should contain a
 *						 column vector that the
 *variables are convolved with.
 *
 * Returns: vector of coefficients for convolution linOp
 */
Tensor get_conv_mat(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == CONV);
  Matrix constant = get_constant_data(*lin.get_linOp_data(), false);
  int rows = lin.get_shape()[0];
  int nonzeros = constant.rows();
  int cols = 1;
  if (lin.get_args()[0]->get_shape().size() > 0) {
      cols = lin.get_args()[0]->get_shape()[0];
  }

  Matrix toeplitz(rows, cols);

  std::vector<Triplet> tripletList;
  tripletList.reserve(nonzeros * cols);
  for (int col = 0; col < cols; col++) {
    int row_start = col;
    for (int k = 0; k < constant.outerSize(); ++k) {
      for (Matrix::InnerIterator it(constant, k); it; ++it) {
        int row_idx = row_start + it.row();
        tripletList.push_back(Triplet(row_idx, col, it.value()));
      }
    }
  }
  toeplitz.setFromTriplets(tripletList.begin(), tripletList.end());
  toeplitz.makeCompressed();
  return build_tensor(toeplitz);
}

/**
 * Return the coefficients for UPPER_TRI: an ENTRIES by ROWS * COLS matrix
 * where the i, j entry in the original matrix has a 1 in row COUNT and
 * corresponding column if j > i and 0 otherwise.
 *
 * Parameters: LinOp with type UPPER_TRI.
 * Returns: vector of coefficients for upper triangular matrix linOp
 */
Tensor get_upper_tri_mat(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == UPPER_TRI);
  int rows = lin.get_args()[0]->get_shape()[0];
  int cols = lin.get_args()[0]->get_shape()[1];

  int entries = lin.get_shape()[0];
  Matrix coeffs(entries, rows * cols);

  std::vector<Triplet> tripletList;
  tripletList.reserve(entries);
  int count = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (j > i) {
        // index in the extracted vector
        int row_idx = count;
        count++;
        // index in the original matrix
        int col_idx = j * rows + i;
        tripletList.push_back(Triplet(row_idx, col_idx, 1.0));
      }
    }
  }
  coeffs.setFromTriplets(tripletList.begin(), tripletList.end());
  coeffs.makeCompressed();
  return build_tensor(coeffs);
}

/**
 * Return the coefficients for DIAG_MAT (diagonal matrix to vector): a
 * N by N^2 matrix where each row has a 1 in the row * N + row entry
 * corresponding to the diagonal and 0 otherwise.
 *
 * Parameters: LinOp of type DIAG_MAT
 *
 * Returns: vector containing coefficient matrix COEFFS
 *
 */
Tensor get_diag_matrix_mat(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == DIAG_MAT);

  int k = lin.get_dense_data()(0, 0);
  int rows = lin.get_shape()[0];
  int original_rows = rows + abs(k);

  Matrix coeffs(rows, original_rows * original_rows);
  std::vector<Triplet> tripletList;
  tripletList.reserve(rows);
  for (int i = 0; i < rows; ++i) {
    // index in the extracted vector
    int row_idx = i;
    // index in the original matrix
    int col_idx;

    if (k == 0) {
      col_idx = i + i * original_rows;
    } else if (k > 0){
      col_idx = i + i * original_rows + (original_rows * k);
    } else {
      col_idx = i + i * original_rows - k;
    }

    tripletList.push_back(Triplet(row_idx, col_idx, 1.0));
  }

  coeffs.setFromTriplets(tripletList.begin(), tripletList.end());
  coeffs.makeCompressed();
  return build_tensor(coeffs);
}

/**
 * Return the coefficients for DIAG_VEC (vector to diagonal matrix): a
 * N^2 by N matrix where each column I has a 1 in row I * N + I
 * corresponding to the diagonal entry and 0 otherwise.
 *
 * Parameters: linOp of type DIAG_VEC
 *
 * Returns: vector containing coefficient matrix COEFFS
 *
 */
Tensor get_diag_vec_mat(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == DIAG_VEC);

  int k = lin.get_dense_data()(0, 0);
  int rows = lin.get_shape()[0];
  int original_rows = rows - abs(k);

  Matrix coeffs(rows*rows, original_rows);
  std::vector<Triplet> tripletList;
  tripletList.reserve(original_rows);
  for (int i = 0; i < original_rows; ++i) {
    // index in the diagonal matrix
    int row_idx;
    // index in the original vector
    int col_idx = i;

    if (k == 0) {
      row_idx = i + i * rows;
    } else if (k > 0) {
      row_idx = i + i * rows + rows * k;
    } else {
      row_idx = i + i * rows - k;
    }
    tripletList.push_back(Triplet(row_idx, col_idx, 1.0));
  }
  coeffs.setFromTriplets(tripletList.begin(), tripletList.end());
  coeffs.makeCompressed();
  return build_tensor(coeffs);
}

/**
 * Return the coefficients for TRANSPOSE: a ROWS*COLS by ROWS*COLS matrix
 * such that element ij in the vectorized matrix is mapped to ji after
 * multiplication (i.e. entry (rows * j + i, i * cols + j) = 1 and else 0)
 *
 * Parameters: linOp of type TRANSPOSE
 *
 * Returns: vector containing coefficient matrix COEFFS
 *
 */
Tensor get_transpose_mat(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == TRANSPOSE);
  int rows = lin.get_shape()[0];
  int cols = lin.get_shape()[1];

  Matrix coeffs(rows * cols, rows * cols);

  std::vector<Triplet> tripletList;
  tripletList.reserve(rows * cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int row_idx = rows * j + i;
      int col_idx = i * cols + j;
      tripletList.push_back(Triplet(row_idx, col_idx, 1.0));
    }
  }
  coeffs.setFromTriplets(tripletList.begin(), tripletList.end());
  coeffs.makeCompressed();
  return build_tensor(coeffs);
}

/**
 * Adds Triplets for slices starting from the current axis.
 *
 * Parameters: triplet list, slices, dimensions, axis, row counter.
 *
 * Returns: new row counter
 *
 */
int add_triplets(std::vector<Triplet> &tripletList,
                 const std::vector<std::vector<int> > &slices,
                 const std::vector<int> &dims, int axis, int col_offset,
                 int row_offset) {
  if (axis < 0) {
    tripletList.push_back(Triplet(row_offset, col_offset, 1.0));
    return row_offset + 1;
  }
  int start = slices[axis][0];
  int end = slices[axis][1];
  int step = slices[axis][2];
  int pointer = start;
  while (true) {
    if (pointer < 0 || pointer >= dims[axis]) {
      break;
    }
    int new_offset = col_offset + pointer * vecprod_before(dims, axis);
    row_offset = add_triplets(tripletList, slices, dims, axis - 1, new_offset,
                              row_offset);
    pointer += step;
    if ((step > 0 && pointer >= end) || (step < 0 && pointer <= end)) {
      break;
    }
  }
  return row_offset;
}

/**
 * Return the coefficients for INDEX: a N by ROWS*COLS matrix
 * where N is the number of total elements in the slice. Element i, j
 * is 1 if element j in the vectorized matrix is the i-th element of the
 * slice and 0 otherwise.
 *
 * Parameters: LinOp of type INDEX
 *
 * Returns: vector containing coefficient matrix COEFFS
 *
 */
Tensor get_index_mat(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == INDEX);
  Matrix coeffs(vecprod(lin.get_shape()),
                vecprod(lin.get_args()[0]->get_shape()));

  /* If slice is empty, return empty matrix */
  if (coeffs.rows() == 0 || coeffs.cols() == 0) {
    return build_tensor(coeffs);
    // Special case for scalars.
  } else if (coeffs.rows() * coeffs.cols() == 1) {
    Matrix coeffs = sparse_eye(1);
    return build_tensor(coeffs);
  }

  /* Set the index coefficients by looping over the column selection
   * first to remain consistent with CVXPY. */
  std::vector<Triplet> tripletList;
  tripletList.reserve(coeffs.rows());
  std::vector<int> dims = lin.get_args()[0]->get_shape();
  assert(lin.get_slice().size() == dims.size());
  add_triplets(tripletList, lin.get_slice(), dims, lin.get_slice().size() - 1,
               0, 0);
  coeffs.setFromTriplets(tripletList.begin(), tripletList.end());
  coeffs.makeCompressed();
  return build_tensor(coeffs);
}

/**
 * Return the coefficients for MUL_ELEM: an N x N diagonal matrix where the
 * n-th element on the diagonal corresponds to the element n = j*rows + i in
 * the data matrix CONSTANT.
 *
 * Parameters: linOp of type MUL_ELEM
 *
 * Returns: vector containing the coefficient matrix COEFFS
 *
 */
Tensor get_mul_elemwise_mat(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == MUL_ELEM);
  // TODO because get_constant_data copies data, this will also
  // copy data ...
  Tensor mul_ten = lin_to_tensor(*lin.get_linOp_data());
  // Convert all the Tensor matrices into diagonal matrices.
  // Replace them in-place.
  for (auto it = mul_ten.begin(); it != mul_ten.end(); ++it) {
    int param_id = it->first;
    const DictMat &var_map = it->second;
    for (auto jit = var_map.begin(); jit != var_map.end(); ++jit) {
      int var_id = jit->first;
      const std::vector<Matrix> &mat_vec = jit->second;
      for (unsigned i = 0; i < mat_vec.size(); ++i) {
        // Diagonalize matrix.
        mul_ten[param_id][var_id][i] = diagonalize(mat_vec[i]);
      }
    }
  }
  return mul_ten;
}

/**
 * Return the coefficients for RMUL (right multiplication): a ROWS * N
 * by COLS * N matrix given by the kronecker product between the
 * transpose of the constant matrix CONSTANT and a N x N identity matrix.
 *
 * Parameters: linOp of type RMUL
 *
 * Returns: vector containing the corresponding coefficient matrix COEFFS
 *
 */
Tensor get_rmul_mat(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == RMUL);
  // Scalar multiplication handled in mul_elemwise.
  assert(lin.get_args()[0]->get_shape().size() > 0);
  Tensor rmul_ten = lin_to_tensor(*lin.get_linOp_data());

  // Interpret as row or column vector as needed.
  if (lin.get_data_ndim() == 1 && lin.get_args()[0]->get_shape()[0] != 1) {
    // Transpose matrices.
    for (auto it = rmul_ten.begin(); it != rmul_ten.end(); ++it) {
      int param_id = it->first;
      const DictMat &var_map = it->second;
      for (auto jit = var_map.begin(); jit != var_map.end(); ++jit) {
        int var_id = jit->first;
        const std::vector<Matrix> &mat_vec = jit->second;
        for (unsigned i = 0; i < mat_vec.size(); ++i) {
          // Transpose matrix.
          rmul_ten[param_id][var_id][i] = mat_vec[i].transpose();
        }
      }
    }
  }
  // Get rows and cols of data (1 if not present).
  int data_rows = (lin.get_linOp_data()->get_shape().size() >= 1)
                      ? lin.get_linOp_data()->get_shape()[0]
                      : 1;
  int data_cols = (lin.get_linOp_data()->get_shape().size() >= 2)
                      ? lin.get_linOp_data()->get_shape()[1]
                      : 1;

  // Interpret as row or column vector as needed.
  int result_rows;
  if (lin.get_args()[0]->get_shape().size() == 0) {
    result_rows = 1;
  } else if (lin.get_args()[0]->get_shape().size() == 1) {
    result_rows = 1;
  } else {
    result_rows = lin.get_args()[0]->get_shape()[0];
  }
  int n = (lin.get_shape().size() > 0) ? result_rows : 1;

  for (auto it = rmul_ten.begin(); it != rmul_ten.end(); ++it) {
    int param_id = it->first;
    const DictMat &var_map = it->second;
    for (auto jit = var_map.begin(); jit != var_map.end(); ++jit) {
      int var_id = jit->first;
      const std::vector<Matrix> &mat_vec = jit->second;
      for (unsigned i = 0; i < mat_vec.size(); ++i) {
        // Form coefficient matrix.
        Matrix coeffs(data_cols * n, data_rows * n);

        std::vector<Triplet> tripletList;
        tripletList.reserve(n * mat_vec[i].nonZeros());
        const Matrix &curr_matrix = mat_vec[i];
        for (int k = 0; k < curr_matrix.outerSize(); ++k) {
          for (Matrix::InnerIterator it(curr_matrix, k); it; ++it) {
            double val = it.value();

            // Data is flattened.
            int col;
            int row;
            if (curr_matrix.rows() == 1) {
              col = it.col() / data_rows;
              row = it.col() % data_rows;
            } else {
              col = it.row() / data_rows;
              row = it.row() % data_rows;
            }
            // Each element of CONSTANT occupies an N x N block in the matrix
            // if X,A in R^{2x2}, then XA yields
            // A_11 0 A_21 0
            // 0 A_11 0 A_21
            // A_12 0 A_22 0
            // 0 A_12 0 A_22
            int row_start = col * n;
            int col_start = row * n;
            for (int i = 0; i < n; ++i) {
              int row_idx = row_start + i;
              int col_idx = col_start + i;
              tripletList.push_back(Triplet(row_idx, col_idx, val));
            }
          }
        }
        coeffs.setFromTriplets(tripletList.begin(), tripletList.end());
        coeffs.makeCompressed();
        // Set block diagonal matrix.
        rmul_ten[param_id][var_id][i].swap(coeffs);
      }
    }
  }
  return rmul_ten;
}

/**
 * Return the coefficients for MUL (left multiplication): a NUM_BLOCKS * ROWS
 * by NUM_BLOCKS * COLS block diagonal matrix where each diagonal block is the
 * constant data BLOCK.
 *
 * Parameters: linOp with type MUL
 *
 * Returns: vector containing coefficient matrix COEFFS
 *
 */
Tensor get_mul_mat(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == MUL);
  // Scalar multiplication handled in mul_elemwise.
  assert(lin.get_args()[0]->get_shape().size() > 0);
  // Get rows and cols of data (1 if not present).
  int data_rows = (lin.get_linOp_data()->get_shape().size() >= 1)
    ? lin.get_linOp_data()->get_shape()[0]
    : 1;
  int data_cols = (lin.get_linOp_data()->get_shape().size() >= 2)
    ? lin.get_linOp_data()->get_shape()[1]
    : 1;

  int num_blocks = (lin.get_args()[0]->get_shape().size() <= 1)
    ? 1
    : lin.get_args()[0]->get_shape()[1];
  // Swap rows and cols if necessary.
  int block_rows = data_rows;
  int block_cols = data_cols;
  if (lin.get_args()[0]->get_shape()[0] != data_cols) {
    block_rows = data_cols;
    block_cols = data_rows;
  }

  const LinOp *data = lin.get_linOp_data();
  Tensor mul_ten;
  bool data_flattened = true;
  if (data->get_type() == SPARSE_CONST || data->get_type() == DENSE_CONST) {
    // Fast path for when data is a sparse matrix.
    // Needed because sparse matrices don't support
    // vectorized views.
    data_flattened = data_rows == 1 || data_cols == 1;
    Matrix coeffs = get_constant_data(*data, false);
    mul_ten = build_tensor(coeffs);
  } else {
    mul_ten = lin_to_tensor(*data);
  }
  // Interpret as row or column vector as needed.
  if (lin.get_data_ndim() == 1 && lin.get_args()[0]->get_shape()[0] != 1) {
    // Transpose matrices.
    for (auto it = mul_ten.begin(); it != mul_ten.end(); ++it) {
      int param_id = it->first;
      const DictMat &var_map = it->second;
      for (auto jit = var_map.begin(); jit != var_map.end(); ++jit) {
        int var_id = jit->first;
        const std::vector<Matrix> &mat_vec = jit->second;
        for (unsigned i = 0; i < mat_vec.size(); ++i) {
          // Transpose matrix.
          // TODO this will copy (unnecessarily?)
          mul_ten[param_id][var_id][i] = mat_vec[i].transpose();
        }
      }
    }
  }

  // TODO may need to speed up. Copying data.
  // Replace every matrix with a block diagonal matrix.
  for (auto it = mul_ten.begin(); it != mul_ten.end(); ++it) {
    int param_id = it->first;
    const DictMat &var_map = it->second;
    for (auto jit = var_map.begin(); jit != var_map.end(); ++jit) {
      int var_id = jit->first;
      const std::vector<Matrix> &mat_vec = jit->second;
      for (unsigned i = 0; i < mat_vec.size(); ++i) {
        // Form block matrix matrix.
        // TODO(akshayka): Fast path for num_blocks=1
        Matrix block_diag(num_blocks * block_rows, num_blocks * block_cols);

        std::vector<Triplet> tripletList;
        tripletList.reserve(num_blocks * mat_vec[i].nonZeros());
        for (int curr_block = 0; curr_block < num_blocks; curr_block++) {
          int start_i = curr_block * block_rows;
          int start_j = curr_block * block_cols;
          const Matrix &curr_matrix = mat_vec[i];
          for (int k = 0; k < curr_matrix.outerSize(); ++k) {
            for (Matrix::InnerIterator it(curr_matrix, k); it; ++it) {
              // Data is flattened.
              int row;
              int col;
              if (data_flattened) {
                if (curr_matrix.rows() == 1) {
                  row = it.col() % block_rows;
                  col = it.col() / block_rows;
                } else {
                  row = it.row() % block_rows;
                  col = it.row() / block_rows;
                }
              } else { // Sparse matrices may not be flattened.
                row = it.row();
                col = it.col();
              }
              tripletList.push_back(
                  Triplet(start_i + row, start_j + col, it.value()));
            }
          }
        }
        block_diag.setFromTriplets(tripletList.begin(), tripletList.end());
        block_diag.makeCompressed();
        // Set block diagonal matrix.
        mul_ten[param_id][var_id][i].swap(block_diag);
      }
    }
  }
  return mul_ten;
}

/**
 * Return the coefficients for PROMOTE: a column vector of size N with all
 * entries 1. Note this is treated as sparse for consistency of later
 * multiplications with sparse matrices.
 *
 * Parameters: linOP with type PROMOTE
 *
 * Returns: vector containing coefficient matrix ONES.
 *
 */
Tensor get_promote_mat(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == PROMOTE);
  int num_entries = vecprod(lin.get_shape());
  Matrix ones = sparse_ones(num_entries, 1);
  ones.makeCompressed();
  return build_tensor(ones);
}

/**
 * Return the coefficients for RESHAPE: a 1x1 matrix [1]. In Eigen, this
 * requires special case handling to multiply against an arbitrary m x n
 * matrix.
 *
 * Parameters: LinOp with type RESHAPE
 *
 * Returns: vector containing the coefficient matrix ONE.
 *
 */
Tensor get_reshape_mat(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == RESHAPE);
  int n = vecprod(lin.get_shape());
  Matrix coeffs = sparse_eye(n);
  coeffs.makeCompressed();
  return build_tensor(coeffs);
}

/**
 * Return the coefficients for DIV: a diagonal matrix where each diagonal
 * entry is 1 / DIVISOR.
 *
 * Parameters: linOp with type DIV
 *
 * Returns: vector containing the coefficient matrix COEFFS
 *
 */
Tensor get_div_mat(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == DIV);
  Matrix constant = get_constant_data(*lin.get_linOp_data(), true);
  int n = constant.rows();

  // build a giant diagonal matrix
  std::vector<Triplet> tripletList;
  tripletList.reserve(n);
  for (int k = 0; k < constant.outerSize(); ++k) {
    for (Matrix::InnerIterator it(constant, k); it; ++it) {
      tripletList.push_back(Triplet(it.row(), it.row(), 1.0 / it.value()));
    }
  }
  Matrix coeffs(n, n);
  coeffs.setFromTriplets(tripletList.begin(), tripletList.end());
  coeffs.makeCompressed();
  return build_tensor(coeffs);
}

/**
 * Return the coefficients for NEG: -I, where I is an identity of size m * n.
 *
 * Parameters: linOp with type NEG
 *
 * Returns: vector containing the coefficient matrix COEFFS
 */
Tensor get_neg_mat(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == NEG);
  int n = vecprod(lin.get_shape());
  Matrix coeffs = sparse_eye(n);
  coeffs *= -1.0;
  coeffs.makeCompressed();
  return build_tensor(coeffs);
}

/**
 * Return the coefficients for TRACE: A single row vector v^T \in R^(n^2)
 * with 1 if v_{i}  corresponds to a diagonal entry (i.e. i * n + i) and 0
 * otherwise.
 *
 * Parameters: LinOp with type TRACE
 *
 * Returns: vector containing the coefficient matrix COEFFS
 *
 */
Tensor get_trace_mat(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == TRACE);
  int rows = lin.get_args()[0]->get_shape()[0];
  std::vector<Triplet> tripletList;
  tripletList.reserve(rows);
  for (int i = 0; i < rows; ++i) {
    tripletList.push_back(Triplet(0, i * rows + i, 1.0));
  }
  Matrix coeffs(1, rows * rows);
  coeffs.setFromTriplets(tripletList.begin(), tripletList.end());
  coeffs.makeCompressed();
  return build_tensor(coeffs);
}

/**
 * Return the coefficient matrix for SUM_ENTRIES. A single row vector of 1's
 * of size 1 by (data.rows x data.cols).
 *
 * Parameters: LinOp with type SUM_ENTRIES
 *
 * Returns: vector containing the coefficient matrix COEFFS
 */
Tensor get_sum_entries_mat(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == SUM_ENTRIES);
  // assumes all args have the same size
  int size = vecprod(lin.get_args()[0]->get_shape());
  Matrix coeffs = sparse_ones(1, size);
  coeffs.makeCompressed();
  return build_tensor(coeffs);
}

/**
 * Return horizontally stacked identity matrices.
 *
 * Parameters: LinOp with type SUM
 *
 * Returns: A vector of length N where each element is a 1x1 matrix
 */
Tensor get_sum_coefficients(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == SUM);
  int n = vecprod(lin.get_shape());
  Matrix coeffs = sparse_eye(n);
  coeffs.makeCompressed();
  return build_tensor(coeffs);
}

/**
 * Return a map from the variable ID to the coefficient matrix for the
 * corresponding VARIABLE linOp, which is an identity matrix of total
 * linop size x total linop size.
 *
 * Parameters: VARIABLE Type LinOp LIN
 *
 * Returns: Map from VARIABLE_ID to coefficient matrix COEFFS for LIN
 *
 */
Tensor get_variable_coeffs(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == VARIABLE);
  int id = get_id_data(lin, arg_idx);

  Tensor ten;
  DictMat &id_to_coeffs = ten[CONSTANT_ID];
  std::vector<Matrix> &mat_vec = id_to_coeffs[id];

  // create a giant identity matrix
  int n = vecprod(lin.get_shape());
  Matrix coeffs = sparse_eye(n);
  coeffs.makeCompressed();
  mat_vec.push_back(Matrix());
  mat_vec[0].swap(coeffs);

  return ten;
}

/**
 * Returns a Tensor with selector matrices
 * for the parameter entries.
 *
 * Parameters: PARAM Type LinOp LIN
 *
 * Returns: Tensor
 *
 */
Tensor get_param_coeffs(const LinOp &lin, int arg_idx) {
  assert(lin.get_type() == PARAM);
  int id = get_id_data(lin, arg_idx);
  // create a giant identity matrix
  unsigned m = (lin.get_shape().size() >= 1) ? lin.get_shape()[0] : 1;
  unsigned n = (lin.get_shape().size() >= 2) ? lin.get_shape()[1] : 1;

  Tensor ten;
  DictMat &dm = ten[id];
  std::vector<Matrix> &mat_vec = dm[CONSTANT_ID];

  // Make mxn matrices with one zero,
  // stack them in column major order.
  for (unsigned j = 0; j < n; ++j) {
    for (unsigned i = 0; i < m; ++i) {
      mat_vec.push_back(sparse_selector(m, n, i, j));
    }
  }
  return ten;
}

/**
 * Returns a map from CONSTANT_ID to the data matrix of the corresponding
 * CONSTANT type LinOp. The coefficient matrix is the data matrix reshaped
 * as a ROWS * COLS by 1 column vector.
 * Note the data is treated as sparse regardless of the underlying
 * representation.
 *
 * Parameters: CONSTANT linop LIN
 *
 * Returns: map from CONSTANT_ID to the coefficient matrix COEFFS for LIN.
 */
Tensor get_const_coeffs(const LinOp &lin, int arg_idx) {
  assert(lin.is_constant());
  Tensor ten;
  DictMat &id_to_coeffs = ten[CONSTANT_ID];
  std::vector<Matrix> &mat_vec = id_to_coeffs[CONSTANT_ID];

  // get coeffs as a column vector
  assert(lin.get_linOp_data() == nullptr);
  // TODO this copies data
  Matrix coeffs = get_constant_data(lin, true);
  coeffs.makeCompressed();
  mat_vec.push_back(Matrix());
  mat_vec[0].swap(coeffs);

  return ten;
}
