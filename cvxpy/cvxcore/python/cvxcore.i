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

%module cvxcore

%{
	#define SWIG_FILE_WITH_INIT
	#include "cvxcore.hpp"
%}

%include "numpy.i"
%include "std_vector.i"
%include "std_map.i"

/* Must call this before using NUMPY-C API */
%init %{
	import_array();
%}

/* Typemap for the addDenseData C++ routine in LinOp.hpp */
%apply (double* IN_FARRAY2, int DIM1, int DIM2) {(double* matrix, int rows, int cols)};

/* Typemap for the addSparseData C++ routine in LinOp.hpp */
%apply (double* INPLACE_ARRAY1, int DIM1) {(double *data, int data_len),
	(double *row_idxs, int rows_len),
	(double *col_idxs, int cols_len)};

%include "LinOp.hpp"
%include "Utils.hpp"

/* Typemap for the getV, getI, getJ, and getConstVec C++ routines in
	 problemData.hpp */
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* values, int num_values)}
%include "ProblemData.hpp"

/* Useful wrappers for the LinOp class */
namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   %template(IntVector2D) vector< vector<int> >;
   %template(DoubleVector2D) vector< vector<double> >;
   %template(IntIntMap) map<int, int>;
   %template(LinOpVector) vector< LinOp * >;
   %template(ConstLinOpVector) vector< const LinOp * >;
}

/* Wrapper for entry point into CVXCanon Library */
ProblemData build_matrix(std::vector< const LinOp* > constraints,
                         int var_length,
                         std::map<int, int> id_to_col,
                         std::map<int, int> param_to_size,
                         int num_threads);
ProblemData build_matrix(std::vector< const LinOp* > constraints,
                         int var_length,
                         std::map<int, int> id_to_col,
                         std::map<int, int> param_to_size,
                         std::vector<int> constr_offsets);
