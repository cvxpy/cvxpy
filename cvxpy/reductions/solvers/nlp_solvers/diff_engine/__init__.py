"""
Copyright 2025, the CVXPY developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
CVXPY integration layer for the DNLP diff engine.

This module converts CVXPY expressions to C expression trees
for automatic differentiation.
"""

from cvxpy.reductions.solvers.nlp_solvers.diff_engine.c_problem import C_problem
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import convert_expr
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.helpers import (
    build_param_dict,
    build_var_dict,
)
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.registry import ATOM_CONVERTERS

__all__ = [
    "C_problem",
    "ATOM_CONVERTERS",
    "build_var_dict",
    "build_param_dict",
    "convert_expr",
]
