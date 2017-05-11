"""
Copyright 2016 Jaehyun Park, 2017 Robin Verschueren

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

import cvxpy
from cvxpy.reductions.reduction import Reduction
from cvxpy.utilities import CoeffExtractor
import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.problems.problem import Problem
from cvxpy.constraints.nonpos import NonPos
from cvxpy.constraints.zero import Zero
from cvxpy.reductions.solution import Solution
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.reductions.inverse_data import InverseData


class QpMatrixStuffing(Reduction):
    """Linearly constrained least squares solver via SciPy.
    """

    def accepts(self, prob):
        """Temporary method to determine whether the given Problem object is suitable for LS solver.
        """
        import cvxpy.constraints.zero as eqc
        import cvxpy.expressions.variables as var
        allowedVariables = (var.variable.Variable, var.symmetric.SymmetricUpperTri)

        # TODO: handle affine objective
        return (
            prob.is_dcp() and
            prob.objective.args[0].is_quadratic() and
            not prob.objective.args[0].is_affine() and
            all([arg.is_affine() for c in prob.constraints for arg in c.args]) and
            all([type(v) in allowedVariables for v in prob.variables()]) and
            all([not v.domain for v in prob.variables()])  # no implicit variable domains
            # (TODO: domains are not implemented yet)
        )

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.
        """
        inverse_data = InverseData(problem)
        # concatenate all variables in one vector
        x = cvxpy.Variable(inverse_data.x_length)

        extractor = CoeffExtractor(inverse_data.var_offsets, inverse_data.x_length)
        objective = problem.objective
        # extract to x.T * P * x + q.T * x + r
        ([P], Q, R) = extractor.quad_form(objective.expr)
        q = np.asarray(Q.todense()).flatten()
        r = R[0]
        new_obj = QuadForm(x, P) + q.T*x + r

        constraints = problem.constraints
        new_cons = []
        ineq_cons = [extractor.get_coeffs(c.expr)[1:] for c in constraints if type(c) == NonPos]
        eq_cons = [extractor.get_coeffs(c.expr)[1:] for c in constraints if type(c) == Zero]
        if ineq_cons:
            A = sp.vstack([C[0] for C in ineq_cons])
            b = np.array([C[1] for C in ineq_cons]).flatten()
            new_cons += [A*x + b <= 0]
        if eq_cons:
            F = sp.vstack([C[0] for C in eq_cons])
            g = np.array([C[1] for C in eq_cons]).flatten()
            new_cons += [F*x + g == 0]

        offset = {s.INEQ_CONSTR: 0, s.EQ_CONSTR: 0}
        inverse_data.new_var_id = x.id
        for c in constraints:
            if type(c) == NonPos:
                inverse_data.cons_id_map[c.constr_id] = (new_cons[0].constr_id, offset[s.INEQ_CONSTR], c.shape)
                offset[s.INEQ_CONSTR] += c.shape[0]*c.shape[1]
            elif type(c) == Zero:
                inverse_data.cons_id_map[c.constr_id] = (new_cons[1].constr_id, offset[s.EQ_CONSTR], c.shape)
                offset[s.EQ_CONSTR] += c.shape[0]*c.shape[1]
            else:
                raise ValueError("Type", type(c), "not allowed in QP")

        new_prob = Problem(cvxpy.Minimize(new_obj), new_cons)
        return (new_prob, inverse_data)

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        if solution.status == s.OPTIMAL:
            primal_vars = dict()
            dual_vars = dict()
            for old_id, tup in inverse_data.cons_id_map.items():
                new_id, offset, shape = tup
                size = shape[0]*shape[1]
                val = solution.dual_vars[new_id][offset:offset+size]
                dual_vars[old_id] = val.reshape(shape, order='F')
            for old_id, tup in inverse_data.id_map.items():
                offset, size = tup
                new_id = inverse_data.new_var_id
                size = shape[0]*shape[1]
                val = solution.primal_vars[new_id][offset:offset+size]
                primal_vars[old_id] = val.reshape(shape, order='F')
            ret = Solution(s.OPTIMAL, solution.opt_val, primal_vars, dual_vars)
        else:
            ret = solution
        return ret
