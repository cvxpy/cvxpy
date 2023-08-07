"""
Copyright 2017 Robin Verschueren

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

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import NonPos, Zero
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.qp2quad_form.qp_matrix_stuffing import (
    ConeDims,
    ParamQuadProg,
)
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.reductions.utilities import group_constraints


class QpSolver(Solver):
    """
    A QP solver interface.
    """
    # Every QP solver supports Zero and NonPos constraints.
    SUPPORTED_CONSTRAINTS = [Zero, NonPos]

    # Some solvers cannot solve problems that do not have constraints.
    # For such solvers, REQUIRES_CONSTR should be set to True.
    REQUIRES_CONSTR = False

    IS_MIP = "IS_MIP"

    def accepts(self, problem):
        return (isinstance(problem, ParamQuadProg)
                and (self.MIP_CAPABLE or not problem.is_mixed_integer())
                and not convex_attributes([problem.x])
                and (len(problem.constraints) > 0 or not self.REQUIRES_CONSTR)
                and all(type(c) in self.SUPPORTED_CONSTRAINTS for c in
                        problem.constraints))

    def _prepare_data_and_inv_data(self, problem):
        data = {}
        inv_data = {self.VAR_ID: problem.x.id}

        constr_map = group_constraints(problem.constraints)
        data[QpSolver.DIMS] = ConeDims(constr_map)
        inv_data[QpSolver.DIMS] = data[QpSolver.DIMS]

        # Add information about integer variables
        inv_data[QpSolver.IS_MIP] = problem.is_mixed_integer()

        data[s.PARAM_PROB] = problem
        return problem, data, inv_data

    def apply(self, problem):
        """
        Construct QP problem data stored in a dictionary.
        The QP has the following form

            minimize      1/2 x' P x + q' x
            subject to    A x =  b
                          F x <= g

        """
        problem, data, inv_data = self._prepare_data_and_inv_data(problem)

        P, q, d, AF, bg = problem.apply_parameters()
        inv_data[s.OFFSET] = d

        # Get number of variables
        n = problem.x.size
        len_eq = data[QpSolver.DIMS].zero
        len_leq = data[QpSolver.DIMS].nonpos

        if len_eq > 0:
            A = AF[:len_eq, :]
            b = -bg[:len_eq]
        else:
            A, b = sp.csr_matrix((0, n)), -np.array([])

        if len_leq > 0:
            F = AF[len_eq:, :]
            g = -bg[len_eq:]
        else:
            F, g = sp.csr_matrix((0, n)), -np.array([])

        # Create dictionary with problem data
        data[s.P] = sp.csc_matrix(P)
        data[s.Q] = q
        data[s.A] = sp.csc_matrix(A)
        data[s.B] = b
        data[s.F] = sp.csc_matrix(F)
        data[s.G] = g
        data[s.BOOL_IDX] = [t[0] for t in problem.x.boolean_idx]
        data[s.INT_IDX] = [t[0] for t in problem.x.integer_idx]
        data['n_var'] = n
        data['n_eq'] = A.shape[0]
        data['n_ineq'] = F.shape[0]

        return data, inv_data
