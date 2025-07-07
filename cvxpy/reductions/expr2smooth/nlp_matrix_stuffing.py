"""
Copyright 2013 Steven Diamond

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
from __future__ import annotations

import numpy as np

import cvxpy.settings as s
from cvxpy.constraints import (
    PSD,
    SOC,
    Equality,
    ExpCone,
    Inequality,
    NonNeg,
    NonPos,
    PowCone3D,
    Zero,
)
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.reductions import InverseData, Solution, cvx_attr2constr
from cvxpy.reductions.matrix_stuffing import (
    MatrixStuffing,
    extract_lower_bounds,
    extract_mip_idx,
    extract_upper_bounds,
)
from cvxpy.reductions.solvers.solving_chain_utils import get_canon_backend
from cvxpy.reductions.utilities import (
    are_args_affine,
    group_constraints,
    lower_equality,
    lower_ineq_to_nonneg,
    nonpos2nonneg,
)
from cvxpy.utilities.coeff_extractor import CoeffExtractor


class NLPMatrixStuffing(MatrixStuffing):
    """Construct matrices for linear cone problems.

    Linear cone problems are assumed to have a linear objective and cone
    constraints which may have zero or more arguments, all of which must be
    affine.
    """
    CONSTRAINTS = 'ordered_constraints'

    def __init__(self, canon_backend: str | None = None):
        self.canon_backend = canon_backend

    def accepts(self, problem):
        valid_obj_curv = problem.objective.expr.is_affine()
        return (type(problem.objective) == Minimize
                and valid_obj_curv
                and not cvx_attr2constr.convex_attributes(problem.variables())
                and are_args_affine(problem.constraints)
                and problem.is_dpp())

    def apply(self, problem):
        pass

    def invert(self, solution, inverse_data):
        """Retrieves a solution to the original problem"""
        var_map = inverse_data.var_offsets
        # Flip sign of opt val if maximize.
        opt_val = solution.opt_val
        if solution.status not in s.ERROR and not inverse_data.minimize:
            opt_val = -solution.opt_val

        primal_vars, dual_vars = {}, {}
        if solution.status not in s.SOLUTION_PRESENT:
            return Solution(solution.status, opt_val, primal_vars, dual_vars,
                            solution.attr)

        # Split vectorized variable into components.
        x_opt = list(solution.primal_vars.values())[0]
        for var_id, offset in var_map.items():
            shape = inverse_data.var_shapes[var_id]
            size = np.prod(shape, dtype=int)
            primal_vars[var_id] = np.reshape(x_opt[offset:offset+size], shape,
                                             order='F')

        return Solution(solution.status, opt_val, primal_vars, dual_vars,
                        solution.attr)
