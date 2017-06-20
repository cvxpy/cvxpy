"""
Copyright 2017 Robin Verschueren

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
import mathprogbasepy as qp
from cvxpy.constraints import NonPos, Zero
from cvxpy.reductions import InverseData, Solution
from cvxpy.solver_interface.conic_solvers.conic_solver import ConicSolver
from cvxpy.solver_interface.reduction_solver import ReductionSolver
from cvxpy.problems.objective import Minimize
from cvxpy.constraints.constraint import Constraint
from cvxpy.problems.objective_attributes import is_qp_objective
from cvxpy.expressions.attributes import is_affine
from cvxpy.constraints.attributes import is_qp_constraint
from cvxpy.problems.problem_analyzer import ProblemAnalyzer


class QpSolver(ReductionSolver):
    """
    A QP solver interface.
    """

    preconditions = {
        (Minimize, is_qp_objective, True),
        (Constraint, is_qp_constraint, True),
        (Constraint, is_affine, True)
    }

    def __init__(self, solver_name):
        self.name = solver_name

    def name(self):
        return self.name

    def import_solver(self):
        import mathprogbasepy as qp
        qp

    def accepts(self, problem):
        return ProblemAnalyzer(problem).matches(self.preconditions)

    def apply(self, problem):
        inverse_data = InverseData(problem)

        obj = problem.objective
        # quadratic part of objective is x.T * P * X but solvers expect 0.5*x.T * P * x.
        P = 2*obj.expr.args[0].args[1].value
        q = obj.expr.args[1].args[0].value.flatten()
        n = P.shape[0]

        ineq_cons = [c for c in problem.constraints if type(c) == NonPos]
        if ineq_cons:
            ineq_coeffs = zip(*[ConicSolver.get_coeff_offset(con.expr) for con in ineq_cons])
            A = sp.vstack(ineq_coeffs[0])
            b = np.concatenate(ineq_coeffs[1])
        else:
            A, b = sp.csr_matrix((0, n)), np.array([])

        eq_cons = [c for c in problem.constraints if type(c) == Zero]
        if eq_cons:
            eq_coeffs = zip(*[ConicSolver.get_coeff_offset(con.expr) for con in eq_cons])
            F = sp.vstack(eq_coeffs[0])
            g = np.concatenate(eq_coeffs[1])
        else:
            F, g = sp.csr_matrix((0, n)), np.array([])

        A = sp.vstack([A, F])
        u = np.concatenate((-b, -g))
        lbA = -np.inf*np.ones(b.shape)
        l = np.concatenate([lbA, -g])

        inverse_data.sorted_constraints = ineq_cons + eq_cons
        return qp.QuadprogProblem(P, q, A, l, u), inverse_data

    def invert(self, solution, inverse_data):
        status = solution.status
        attr = {s.SOLVE_TIME: solution.cputime}

        if status in s.SOLUTION_PRESENT:
            opt_val = solution.obj_val
            primal_vars = {inverse_data.id_map.keys()[0]: np.array(solution.x)}
            dual_vars = ConicSolver.get_dual_values(solution.y, inverse_data.sorted_constraints)
            attr[s.NUM_ITERS] = solution.total_iter
        else:
            primal_vars = None
            dual_vars = None
            opt_val = np.inf
            if status == s.UNBOUNDED:
                opt_val = -np.inf
        return Solution(status, opt_val, primal_vars, dual_vars, attr)

    def solve(self, problem, warm_start, verbose, solver_opts):
        data, inverse_data = self.apply(problem)
        solution = data.solve(solver=self.name, verbose=verbose)
        return self.invert(solution, inverse_data)
