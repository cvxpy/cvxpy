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
from cvxpy.atoms import QuadForm, reshape
from cvxpy.problems.problem import Problem
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.matrix_stuffing import MatrixStuffing
from cvxpy.reductions.qp2quad_form.qp2symbolic_qp import Qp2SymbolicQp
from cvxpy.utilities.coeff_extractor import CoeffExtractor
from cvxpy.problems.objective import Minimize


class QpMatrixStuffing(MatrixStuffing):
    """Fills in numeric values for this problem instance.
    """

    def accepts(self, prob):
        import cvxpy.expressions.variables as var
        allowedVariables = (var.variable.Variable, var.symmetric.SymmetricUpperTri)

        return (
            prob.is_dcp() and
            prob.objective.args[0].is_quadratic() and
            all([arg.is_affine() for c in prob.constraints for arg in c.args]) and
            all([type(v) in allowedVariables for v in prob.variables()]) and
            all([not v.domain for v in prob.variables()])  # no implicit variable domains
            # (TODO: domains are not implemented yet)
        )

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution."""
        qp, inverse_data_stack = Qp2SymbolicQp().apply(problem)

        if not self.accepts(qp):
            raise ValueError("This QP can not be stuffed")

        inverse_data = InverseData(qp)
        extractor = CoeffExtractor(inverse_data)

        # extract to x.T * P * x + q.T * x + r
        (P, q, r) = extractor.quad_form(qp)

        # concatenate all variables in one vector
        x = cvxpy.Variable(inverse_data.x_length)
        new_obj = QuadForm(x, P) + q.T*x + r

        constraints = qp.constraints
        new_cons = []
        for constraint in constraints:
            assert len(constraint.args) == 1
            A, b = extractor.affine(constraint.expr)
            arg = reshape(A*x+b, constraint.args[0].shape)
            new_con = type(constraint)(arg)
            new_cons += [new_con]
            inverse_data.cons_id_map[constraint.id] = new_con.id

        inverse_data.minimize = type(problem.objective) == Minimize
        new_prob = Problem(cvxpy.Minimize(new_obj), new_cons)
        inverse_data_stack.append(inverse_data)
        return new_prob, inverse_data_stack

    def invert(self, solution, inverse_data_stack):
        """Returns the solution to the original problem given the inverse_data.
        """
        inv = inverse_data_stack.pop()
        solution = MatrixStuffing().invert(solution, inv)
        return Qp2SymbolicQp().invert(solution, inverse_data_stack)
