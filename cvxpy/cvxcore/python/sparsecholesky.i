//   Copyright 2023, the CVXPY Authors
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

%module sparsecholesky

%{
	#define SWIG_FILE_WITH_INIT
	#include "sparsecholesky.hpp"
%}

%include "numpy.i"
%include "std_vector.i"
%include "std_string.i"
%include "exception.i"

namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
}

%exception {
  try {
    $action
  } catch (const CholeskyFailure& e) {
     PyErr_SetString(SWIG_Python_ExceptionType(SWIGTYPE_p_CholeskyFailure), e.what());
     SWIG_fail;
  } catch(const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  } catch(...) {
    SWIG_exception(SWIG_UnknownError, "");
  }
}

%exceptionclass CholeskyFailure;

struct CholeskyFailure : public std::runtime_error {
  CholeskyFailure(const std::string& msg) : std::runtime_error{msg} {}
};


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


