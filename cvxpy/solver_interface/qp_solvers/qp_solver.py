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

import gurobipy as grb
import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import NonPos, Zero
from cvxpy.reductions.qp2quad_form.qp_matrix_stuffing import QpMatrixStuffing
from cvxpy.reductions.solution import Solution
from cvxpy.solver_interface.conic_solvers.conic_solver import ConicSolver
from cvxpy.solver_interface.reduction_solver import ReductionSolver


class QpSolver(ReductionSolver):
    """
    A QP solver interface.
    """

    def __init__(self, solver_name):
        self.name = solver_name

    def name(self):
        return self.name

    def import_solver(self):
        import mathprogbasepy as qp
        qp

    def accepts(self, problem):
        return problem.is_qp()

    def apply(self, problem):
        stuffed_problem, inverse_data_stack = QpMatrixStuffing().apply(problem)
        if not self.accepts(stuffed_problem):
            raise ValueError("QP solver reduction is not applicable to problem")

        import mathprogbasepy as qp
        obj = stuffed_problem.objective
        eq = [c for c in stuffed_problem.constraints if type(c) == Zero]
        ineq = [c for c in stuffed_problem.constraints if type(c) == NonPos]

        P = 2*obj.expr.args[0].args[1].value
        q = obj.expr.args[1].args[0].value.flatten()
        n = P.shape[0]
        inverse_data = {self.VAR_ID: stuffed_problem.variables()[0].id}
        if ineq:
            inverse_data[self.NEQ_CONSTR] = ineq[0].id
            A, b = ConicSolver.get_coeff_offset(ineq[0].expr)
        else:
            A, b = sp.csr_matrix((0, n)), np.array([])
        if eq:
            inverse_data[self.EQ_CONSTR] = eq[0].id
            F, g = ConicSolver.get_coeff_offset(eq[0].expr)
        else:
            F, g = sp.csr_matrix((0, n)), np.array([])
        A = sp.vstack([A, F])
        u = np.concatenate((-b, -g))
        lbA = -grb.GRB.INFINITY*np.ones(b.shape)
        l = np.concatenate([lbA, -g])

        inverse_data['ineq_offset'] = b.shape[0]
        inverse_data_stack.append(inverse_data)
        return qp.QuadprogProblem(P, q, A, l, u), inverse_data_stack

    def invert(self, solution, inverse_data_stack):
        inverse_data = inverse_data_stack.pop()
        status = solution.status
        cputime = solution.cputime
        attr = {s.SOLVE_TIME: cputime}
        if status in s.SOLUTION_PRESENT:
            opt_val = solution.obj_val
            primal_vars = {inverse_data[self.VAR_ID]: np.array(solution.x)}
            dual_vars = {}
            if self.NEQ_CONSTR in inverse_data:
                ineq_constr_id = inverse_data[self.NEQ_CONSTR]
                dual_vars[ineq_constr_id] = solution.y[:inverse_data['ineq_offset']]
            if self.EQ_CONSTR in inverse_data:
                eq_constr_id = inverse_data[self.EQ_CONSTR]
                dual_vars[eq_constr_id] = solution.y[inverse_data['ineq_offset']:]
            total_iter = solution.total_iter
            attr[s.NUM_ITERS] = total_iter
        else:  # no solution
            primal_vars = None
            dual_vars = None
            opt_val = np.inf
            if status == s.UNBOUNDED:
                opt_val = -np.inf
        sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        return QpMatrixStuffing().invert(sol, inverse_data_stack)

    def solve(self, problem, warm_start, verbose, solver_opts):
        data, inverse_data = self.apply(problem)
        solution = data.solve(solver=self.name, verbose=verbose)
        return self.invert(solution, inverse_data)
