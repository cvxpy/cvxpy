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

#include <vector>
#include <cassert>
#include <iostream>
#include "Utils.hpp"

/* ID for all coefficient matrices associated with linOps of CONSTANT_TYPE */
static const int CONSTANT_ID = -1;

/* TYPE of each LinOP */
enum operatortype {
	VARIABLE,
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
	KRON
};

/* linOp TYPE */
typedef operatortype OperatorType;

/* LinOp Class mirrors the CVXPY linOp class. Data fields are determined
 	 by the TYPE of LinOp. No error checking is performed on the data fields,
 	 and the semantics of SIZE, ARGS, and DATA depends on the linop TYPE. */
class LinOp {
public:
	OperatorType type;
	std::vector<int> size;
	/* Children LinOps in the tree */
	std::vector<LinOp*> args;

  /* Dimensions of data */
  int data_ndim;
	/* Sparse Data Fields */
	bool sparse; // True only if linOp has sparse_data
	Matrix sparse_data;

	/* Dense Data Field */
	Eigen::MatrixXd dense_data;

	/* Slice Data: stores slice data as (row_slice, col_slice)
	 * where slice = (start, end, step_size) */
	std::vector<std::vector<int> > slice;

	/* Constructor */
	LinOp(){
		sparse = false; // sparse by default
	}

	/* Checks if LinOp is constant type */
	bool has_constant_type() {
		return  type == SCALAR_CONST || type == DENSE_CONST
		        || type == SPARSE_CONST;
	}

	/* Initializes DENSE_DATA. MATRIX is a pointer to the data of a 2D
	 * numpy array, ROWS and COLS are the size of the ARRAY.
	 *
	 * MATRIX must be a contiguous array of doubles aligned in fortran
	 * order.
	 *
	 * NOTE: The function prototype must match the type-map in CVXCanon.i
	 * exactly to compile and run properly.
	 */
	void set_dense_data(double* matrix, int rows, int cols) {
		dense_data = Eigen::Map<Eigen::MatrixXd> (matrix, rows, cols);
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
	                     int rows_len, double *col_idxs, int cols_len,
	                     int rows, int cols) {

		assert(rows_len == data_len && cols_len == data_len);
		sparse = true;
		Matrix sparse_coeffs(rows, cols);
		std::vector<Triplet> tripletList;
		tripletList.reserve(data_len);
		for (int idx = 0; idx < data_len; idx++) {
			tripletList.push_back(Triplet(int(row_idxs[idx]), int(col_idxs[idx]),
			                              data[idx]));
		}
		sparse_coeffs.setFromTriplets(tripletList.begin(), tripletList.end());
		sparse_coeffs.makeCompressed();
		sparse_data = sparse_coeffs;
    data_ndim = 2;
	}
};
#endif
