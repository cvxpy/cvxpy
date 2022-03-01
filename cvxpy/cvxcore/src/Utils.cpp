#include "Utils.hpp"
#include <functional>
#include <iostream>
#include <numeric>

int vecprod(const std::vector<int> &vec) {
  return std::accumulate(vec.begin(), vec.end(), 1.0, std::multiplies<int>());
}

int vecprod_before(const std::vector<int> &vec, int end) {
  return std::accumulate(vec.rbegin() + vec.size() - end, vec.rend(), 1.0,
                         std::multiplies<int>());
}

// TODO: Several loops in this file are trivially parallelizable;
//       consider parallelizing some (parallelism will be nested inside
//       build_matrix)
//
// multiply two vectors of matrices
// get a new vector of matrices
std::vector<Matrix> mat_vec_mul(const std::vector<Matrix> &lh_vec,
                                const std::vector<Matrix> &rh_vec) {
  // Can only have single matrix * many, not many * many.
  assert(lh_vec.size() == 1 || rh_vec.size() == 1);
  std::vector<Matrix> result;
  result.reserve(lh_vec.size() * rh_vec.size());
  for (unsigned i = 0; i < lh_vec.size(); ++i) {
    for (unsigned j = 0; j < rh_vec.size(); ++j) {
      // prune away explicit nonzeros
      result.push_back(lh_vec[i] * rh_vec[j]);
    }
  }
  return result;
}

// Accumulate right hand vector of matrices
// into left hand by addition.
void acc_mat_vec(std::vector<Matrix> &lh_mat_vec,
                 const std::vector<Matrix> &rh_mat_vec) {
  // Length of vectors must match.
  assert(lh_mat_vec.size() == rh_mat_vec.size());
  for (unsigned i = 0; i < rh_mat_vec.size(); ++i) {
    lh_mat_vec[i] += rh_mat_vec[i];
  }
}

// What is multiplication?
//    for mat in rh_map:
//         multiply rh_map by mat to get new vector.
//    stack the vectors.
//    under key key1*key2 (constant * key = key)
DictMat dict_mat_mul(const DictMat &lh_dm, const DictMat &rh_dm) {
  DictMat result;
  for (auto it = lh_dm.begin(); it != lh_dm.end(); ++it) {
    // Left hand is always constant.
    assert(it->first == CONSTANT_ID);
    const std::vector<Matrix> &lh_mat_vec = it->second;
    for (auto jit = rh_dm.begin(); jit != rh_dm.end(); ++jit) {
      int rh_var_id = jit->first;
      const std::vector<Matrix> &rh_mat_vec = jit->second;
      if (result.count(rh_var_id) == 0) {
        result[rh_var_id] = mat_vec_mul(lh_mat_vec, rh_mat_vec);
      } else { // Sum matrices.
        acc_mat_vec(result[rh_var_id], mat_vec_mul(lh_mat_vec, rh_mat_vec));
      }
    }
  }
  return result;
}

// Accumulate right hand DictMat
// into left hand by addition.
void acc_dict_mat(DictMat &lh_dm, const DictMat &rh_dm) {
  for (auto it = rh_dm.begin(); it != rh_dm.end(); ++it) {
    int rh_var_id = it->first;
    const std::vector<Matrix> &rh_mat_vec = it->second;
    if (lh_dm.count(rh_var_id) == 0) {
      // TODO swap?
      lh_dm[rh_var_id] = rh_mat_vec;
    } else { // Accumulate into lh_dm vector<Matrix>.
      acc_mat_vec(lh_dm[rh_var_id], rh_mat_vec);
    }
  }
}

// What is multiplication?
// for key1 in map1 by key2 in map2:
//    product map add map1[key1] * map2[key2]
//    for mat in map2[key2]:
//         multiply map1[key1] by mat to get new vector.
//    stack the vectors.
//    under key key1*key2 (constant * key = key)
Tensor tensor_mul(const Tensor &lh_ten, const Tensor &rh_ten) {
  Tensor result;
  for (auto it = lh_ten.begin(); it != lh_ten.end(); ++it) {
    int lh_param_id = it->first;
    const DictMat &lh_var_map = it->second;
    for (auto jit = rh_ten.begin(); jit != rh_ten.end(); ++jit) {
      int rh_param_id = jit->first;
      const DictMat &rh_var_map = jit->second;
      // No cross terms allowed.
      assert(lh_param_id == CONSTANT_ID || rh_param_id == CONSTANT_ID);
      int cross_id;
      if (lh_param_id == CONSTANT_ID) {
        cross_id = rh_param_id;
      } else {
        cross_id = lh_param_id;
      }
      if (result.count(cross_id) == 0) {
        result[cross_id] = dict_mat_mul(lh_var_map, rh_var_map);
      } else { // Accumulate cross_id DictMat.
        acc_dict_mat(result[cross_id], dict_mat_mul(lh_var_map, rh_var_map));
      }
    }
  }
  return result;
}

// Accumulate right hand Tensor into left hand by addition.
void acc_tensor(Tensor &lh_ten, const Tensor &rh_ten) {
  for (auto it = rh_ten.begin(); it != rh_ten.end(); ++it) {
    int rh_param_id = it->first;
    const DictMat &rh_dm = it->second;
    if (lh_ten.count(rh_param_id) == 0) {
      lh_ten[rh_param_id] = rh_dm;
    } else { // Accumulate into lh_dm vector<Matrix>.
      acc_dict_mat(lh_ten[rh_param_id], rh_dm);
    }
  }
}

// Create a giant diagonal matrix with entries given by the entries of mat.
Matrix diagonalize(const Matrix &mat) {
  Matrix diag(mat.rows() * mat.cols(), mat.rows() * mat.cols());
  std::vector<Triplet> tripletList;
  tripletList.reserve(mat.nonZeros());
  for (int k = 0; k < mat.outerSize(); ++k) {
    for (Matrix::InnerIterator it(mat, k); it; ++it) {
      unsigned loc = it.row() + k * mat.rows();
      tripletList.push_back(Triplet(loc, loc, it.value()));
    }
  }
  diag.setFromTriplets(tripletList.begin(), tripletList.end());
  diag.makeCompressed();
  return diag;
}
