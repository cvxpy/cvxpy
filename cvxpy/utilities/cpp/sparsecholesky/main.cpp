// Copyright, the CVXPY authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include "../../../cvxcore/include/Eigen/Core"
#include "../../../cvxcore/include/Eigen/Sparse"
#include "../../../cvxcore/include/Eigen/SparseCholesky"
#include <vector>
#include <string>

PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#define NULL_MATRIX Eigen::SparseMatrix<double>(0, 0)
typedef Eigen::SparseMatrix<double> Matrix;
typedef Eigen::Triplet<double> Triplet;

struct CholeskyFailure : public std::runtime_error {
  CholeskyFailure(const std::string& msg) : std::runtime_error{msg} {}
};

Matrix sparse_from_vecs(
    int n_rows,
    int n_cols,
    std::vector<int> &rows,
    std::vector<int> &cols,
    std::vector<double> vals
) {
    int size = rows.size();
    std::vector<Triplet> trl;
    trl.reserve(size);
    for (int i = 0; i < size; ++i) {
        trl.push_back(Triplet(rows[i], cols[i], vals[i]));
    }
    Matrix mat(n_rows, n_cols);
    mat.setFromTriplets(trl.begin(), trl.end());
    return mat;
}

void vecs_from_sparse(
    Matrix mat,
    std::vector<int> &rows,
    std::vector<int> &cols,
    std::vector<double> &vals
) {
    rows.clear();
    cols.clear();
    vals.clear();
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (Matrix::InnerIterator it(mat, k); it; ++it) {
            vals.push_back(it.value());
            rows.push_back(it.row());
            cols.push_back(it.col());
        }
    }
    return;
}

void sparse_chol_from_vecs(
    int n,
    std::vector<int> &in_rows,
    std::vector<int> &in_cols,
    std::vector<double> &in_vals,
    std::vector<int> &out_pivots,
    std::vector<int> &out_rows,
    std::vector<int> &out_cols,
    std::vector<double> &out_vals
) {
    // read in data
    Matrix mat = sparse_from_vecs(n, n, in_rows, in_cols, in_vals);

    // compute the sparse Cholesky decomposition
    Eigen::SimplicialLLT<Matrix> cholobj(mat);
    auto status = cholobj.info();
    if (status == Eigen::ComputationInfo::NumericalIssue) {
        throw CholeskyFailure{"Cholesky decomposition failed."};
    }
    Matrix L = cholobj.matrixL();
    Eigen::PermutationMatrix<Eigen::Dynamic> P = cholobj.permutationP();
    auto p = P.indices();

    // Prepare output data
    out_pivots.clear(); out_pivots.reserve(n);
    int nnz = L.nonZeros();
    out_rows.clear(); out_rows.reserve(nnz);
    out_cols.clear(); out_cols.reserve(nnz);
    out_vals.clear(); out_vals.reserve(nnz);

    // write output
    for (int i = 0; i < n; ++i)
        out_pivots.push_back(p.data()[i]);
    L.makeCompressed();
    vecs_from_sparse(L, out_rows, out_cols, out_vals);
    return;
}


namespace py = pybind11;

PYBIND11_MODULE(_cvxpy_sparsecholesky, m) {
    py::bind_vector<std::vector<int>>(m, "IntVector");
    py::bind_vector<std::vector<double>>(m, "DoubleVector");

    m.def("sparse_chol_from_vecs", &sparse_chol_from_vecs, R"pbdoc(
        Compute a sparse cholesky decomposition from COO-format data.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
