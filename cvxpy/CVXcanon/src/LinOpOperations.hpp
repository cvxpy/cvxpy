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

#ifndef LINOPOPERATIONS_H
#define LINOPOPERATIONS_H

#include <vector>
#include <map>
#include "Utils.hpp"
#include "LinOp.hpp"

std::map<int, Matrix> get_variable_coeffs(LinOp &lin);
std::map<int, Matrix> get_const_coeffs(LinOp &lin);
std::vector<Matrix> get_func_coeffs(LinOp& lin);

#endif
