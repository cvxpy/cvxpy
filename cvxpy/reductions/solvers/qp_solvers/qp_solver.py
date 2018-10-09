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

from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.constraints import NonPos, Zero
from cvxpy.problems.objective import Minimize
from cvxpy.reductions import InverseData
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.reductions.utilities import are_args_affine
import cvxpy.settings as s


def is_stuffed_qp_objective(objective):
    """QPSolver requires objectives to be stuffed in the following way.
    """
    expr = objective.expr
    return (type(expr) == AddExpression
            and len(expr.args) == 2
            and type(expr.args[0]) == QuadForm
            and type(expr.args[1]) == MulExpression
            and expr.args[1].is_affine())


class QpSolver(Solver):
    """
    A QP solver interface.
    """

    def accepts(self, problem):
        return (type(problem.objective) == Minimize
                and is_stuffed_qp_objective(problem.objective)
                and all(type(c) == Zero or type(c) == NonPos
                        for c in problem.constraints)
                and are_args_affine(problem.constraints))

    def apply(self, problem):
        """
        Construct QP problem data stored in a dictionary.
        The QP has the following form

            minimize      1/2 x' P x + q' x
            subject to    A x =  b
                          F x <= g

        """
        inverse_data = InverseData(problem)

        obj = problem.objective
        # quadratic part of objective is x.T * P * x but solvers expect
        # 0.5*x.T * P * x.
        P = 2*obj.expr.args[0].args[1].value
        q = obj.expr.args[1].args[0].value.flatten()

        # Get number of variables
        n = problem.size_metrics.num_scalar_variables

        # TODO(akshayka): This dependence on ConicSolver is hacky; something
        # should change here.
        eq_cons = [c for c in problem.constraints if type(c) == Zero]
        if eq_cons:
            eq_coeffs = list(zip(*[ConicSolver.get_coeff_offset(con.expr)
                                   for con in eq_cons]))
            A = sp.vstack(eq_coeffs[0])
            b = - np.concatenate(eq_coeffs[1])
        else:
            A, b = sp.csr_matrix((0, n)), -np.array([])

        ineq_cons = [c for c in problem.constraints if type(c) == NonPos]
        if ineq_cons:
            ineq_coeffs = list(zip(*[ConicSolver.get_coeff_offset(con.expr)
                                     for con in ineq_cons]))
            F = sp.vstack(ineq_coeffs[0])
            g = - np.concatenate(ineq_coeffs[1])
        else:
            F, g = sp.csr_matrix((0, n)), -np.array([])

        # Create dictionary with problem data
        variables = problem.variables()[0]
        data = {}
        data[s.P] = sp.csc_matrix(P)
        data[s.Q] = q
        data[s.A] = sp.csc_matrix(A)
        data[s.B] = b
        data[s.F] = sp.csc_matrix(F)
        data[s.G] = g
        data[s.BOOL_IDX] = [t[0] for t in variables.boolean_idx]
        data[s.INT_IDX] = [t[0] for t in variables.integer_idx]
        data['n_var'] = n
        data['n_eq'] = A.shape[0]
        data['n_ineq'] = F.shape[0]

        inverse_data.sorted_constraints = eq_cons + ineq_cons

        # Add information about integer variables
        inverse_data.is_mip = \
            len(data[s.BOOL_IDX]) > 0 or len(data[s.INT_IDX]) > 0

        return data, inverse_data
