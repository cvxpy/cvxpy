"""
Copyright 2016 Jaehyun Park

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
from cvxpy.utilities import QuadCoeffExtractor
import numpy as np
import scipy.sparse as sp
from cvxpy.reductions.solution import OPTIMAL
from cvxpy.reductions.solution import Solution

class QPMatrixStuffing(Reduction):
    """Linearly constrained least squares solver via SciPy.
    """

    def accepts(self, problem):
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
            all([c._expr.is_affine() for c in prob.constraints]) 1and
            all([type(v) in allowedVariables for v in prob.variables()]) and
            all([not v.domain for v in prob.variables()])  # no implicit variable domains
            # (TODO: domains are not implemented yet)
        )

    def get_inverse_data(self, objective, constraints, cached_data=None):
        class InverseData(object):
            def __init__(self, objective, constraints):
                vars_ = objective.variables()
                for c in constraints:
                    vars_ += c.variables()
                vars_ = list(set(vars_))
                self.vars_ = vars_
                self.id_map, self.var_offsets, self.x_length = self.get_var_offsets(vars_)
                self.cons_id_map = dict()

            def get_var_offsets(self, variables):
                id_map = {}
                var_offsets = {}
                vert_offset = 0
                for x in variables:
                    var_offsets[x.id] = vert_offset
                    vert_offset += x.size[0]*x.size[1]
                    id_map[x.id] = (vert_offset, x.size)
                return (id_map, var_offsets, vert_offset)

        return InverseData(objective, constraints)

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.
        """
        objective = problem.objective
        constraints = problem.constraints

        inverse_data = self.get_inverse_data(objective, constraints)

        extractor = QuadCoeffExtractor(inverse_data.var_offsets, inverse_data.x_length)

        # Extract the coefficients
        (Ps, Q, R) = extractor.get_coeffs(objective.args[0])

        P = Ps[0]
        q = np.asarray(Q.todense()).flatten()
        r = R[0]

        x = cvxpy.Variable(inverse_data.x_length)
        new_obj = cvxpy.quad_form(x, P) + q.T*x + r
        new_cons = []

        ineq_cons = [extractor.get_coeffs(c._expr)[1:] for c in constraints if c.OP_NAME == "<="]
        eq_cons = [extractor.get_coeffs(c._expr)[1:] for c in constraints if c.OP_NAME == "=="]
        A = sp.vstack([C[0] for C in ineq_cons])
        b = np.array([C[1] for C in ineq_cons]).flatten()
        F = sp.vstack([C[0] for C in eq_cons])
        g = np.array([C[1] for C in eq_cons]).flatten()
        new_cons = [A*x + b <= 0, F*x + g == 0]

        cnts = [0, 0]
        inverse_data.new_var_id = x.id
        for c in constraints:
            if c.OP_NAME == "<=":
                inverse_data.cons_id_map[c.constr_id] = (new_cons[0].constr_id, cnts[0], c._expr.shape)
                cnts[0] += c._expr.shape[0]*c._expr.shape[1]
            else:
                inverse_data.cons_id_map[c.constr_id] = (new_cons[1].constr_id, cnts[1], c._expr.shape)
                cnts[1] += c._expr.shape[0]*c._expr.shape[1]

        new_prob = cvx.Minimize(new_obj, new_cons)

        return (new_prob, inverse_data)


    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        if solution.status == OPTIMAL:
            primal_vars = dict()
            dual_vars = dict()
            for (old_id, tup) in inverse_data.cons_id_map:
                (new_id, offset, shape) = tup
                size = shape[0]*shape[1]
                val = solution.dual_vars[new_id][offset:offset+size]
                dual_vars[old_id] = val.reshape(shape, order='F')
            for (old_id, tup) in inverse_data.id_map:
                (offset, size) = tup
                new_id = inverse_data.new_var_id
                size = shape[0]*shape[1]
                val = solution.primal_vars[new_id][offset:offset+size]
                primal_vars[old_id] = val.reshape(shape, order='F')
            ret = Solution(OPTIMAL, solution.opt_val, primal_vars, dual_vars)
        else:
            ret = solution
        return ret
