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

#ifndef CVXCANON_H
#define CVXCANON_H
#define EIGEN_MPL2_ONLY
#include "LinOp.hpp"
#include "ProblemData.hpp"
#include "Utils.hpp"
#include <vector>

typedef std::map<int, std::vector<ProblemData> > ProblemTensor;

// Top Level Entry point
ProblemData build_matrix(std::vector<const LinOp *> constraints, int var_length,
                         std::map<int, int> id_to_col,
                         std::map<int, int> param_to_size,
                         int num_threads);
ProblemData build_matrix(std::vector<const LinOp *> constraints, int var_length,
                         std::map<int, int> id_to_col,
                         std::map<int, int> param_to_size,
                         std::vector<int> constr_offsets);
#endif
