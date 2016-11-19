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

import cvxpy.settings as s
from cvxpy.utilities import QuadCoeffExtractor
import numpy as np
import scipy.sparse as sp

class QPMatrixStuffing(Reduction):
    """Linearly constrained least squares solver via SciPy.
    """

    def accepts(self, problem):
        """Temporary method to determine whether the given Problem object is suitable for LS solver.
        """
        import cvxpy.constraints.eq_constraint as eqc
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

    def get_sym_data(self, objective, constraints, cached_data=None):
        class SymData(object):
            def __init__(self, objective, constraints):
                self.constr_map = {s.EQ: constraints}
                vars_ = objective.variables()
                for c in constraints:
                    vars_ += c.variables()
                vars_ = list(set(vars_))
                self.vars_ = vars_
                self.var_offsets, self.var_sizes, self.x_length = self.get_var_offsets(vars_)

            def get_var_offsets(self, variables):
                var_offsets = {}
                var_sizes = {}
                vert_offset = 0
                for x in variables:
                    var_sizes[x.id] = x.size
                    var_offsets[x.id] = vert_offset
                    vert_offset += x.size[0]*x.size[1]
                return (var_offsets, var_sizes, vert_offset)

        return SymData(objective, constraints)

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.
        """
        objective = problem.objective
        constraints = problem.constraints

        sym_data = self.get_sym_data(objective, constraints)

        id_map = sym_data.var_offsets
        N = sym_data.x_length

        extractor = QuadCoeffExtractor(id_map, N)

        # Extract the coefficients
        (Ps, Q, R) = extractor.get_coeffs(objective.args[0])

        P = Ps[0]
        q = np.asarray(Q.todense()).flatten()
        r = R[0]

        x = cvxpy.Variable(N)
        new_obj = cvxpy.quad_form(x, P) + q.T*x + r
        new_cons = []

        ineq_cons = [extractor.get_coeffs(c._expr)[1:] for c in constraints if c.OP_NAME == "<="]
        eq_cons = [extractor.get_coeffs(c._expr)[1:] for c in constraints if c.OP_NAME == "=="]
        A = sp.vstack([C[0] for C in ineq_cons])
        b = np.array([C[1] for C in ineq_cons]).flatten()
        F = sp.vstack([C[0] for C in eq_cons])
        g = np.array([C[1] for C in eq_cons]).flatten()

        new_cons = [A*x + b <= 0, F*x + g == 0]
        new_prob = cvx.Minimize(new_obj, new_cons)

        return (new_prob, sym_data)


    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """

# primal_vars: dict of id to numpy ndarray
# dual_vars: dict of id to numpy ndarray
# opt_val:
# status
        id_map = inverse_data.var_offsets
        N = inverse_data.x_length

        x = solution.primal_vars
        lmb = solution.dual_vars[0]
        nu = solution.dual_vars[1]
        status = solution.status
        ret = {
            "primal_vars": None,
            "dual_vars": None,
            "opt_val": None,
            "status": status
        }
        if status == "Optimal":
            ret.primal_vars = x
            ret.dual_vars = (lmb, nu)
        return ret
