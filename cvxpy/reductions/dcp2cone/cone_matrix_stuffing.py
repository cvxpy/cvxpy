"""
Copyright 2013 Steven Diamond

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

from cvxpy.atoms import reshape
from cvxpy.expressions.variables import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.matrix_stuffing import MatrixStuffing
from cvxpy.utilities.coeff_extractor import CoeffExtractor


class ConeMatrixStuffing(MatrixStuffing):
    """Construct matrices for linear cone problems.

    Linear cone problems are assumed to have a linear objective and cone
    constraints which may have zero or more arguments, all of which must be
    affine.

    minimize   c'x
    subject to cone_constr1(A_1*x + b_1, ...)
               ...
               cone_constrK(A_i*x + b_i, ...)
    """

    def accepts(self, problem):
        return (
            problem.is_dcp() and
            problem.objective.args[0].is_affine() and
            all([arg.is_affine() for c in problem.constraints for arg in c.args])
        )

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution."""
        objective = problem.objective
        constraints = problem.constraints

        inverse_data = InverseData(problem)
        N = inverse_data.x_length

        extractor = CoeffExtractor(inverse_data)

        # Extract the coefficients
        C, R = extractor.get_coeffs(objective.args[0])
        c = np.asarray(C.todense()).flatten()
        r = R[0]
        x = Variable(N)
        if type(objective) == Minimize:
            new_obj = c.T*x + r
        else:
            new_obj = (-c).T*x + -r
        # Form the constraints
        new_cons = []
        for con in constraints:
            arg_list = []
            for arg in con.args:
                A, b = extractor.get_coeffs(arg)
                arg_list.append(reshape(A*x + b, arg.shape))
            new_cons.append(type(con)(*arg_list))
            inverse_data.cons_id_map[con.id] = new_cons[-1].id

        # Map of old constraint id to new constraint id.
        inverse_data.minimize = type(problem.objective) == Minimize
        new_prob = Problem(Minimize(new_obj), new_cons)
        return new_prob, inverse_data
