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

from cvxpy.constraints import (Equality, ExpCone, Inequality,
                               SOC, Zero, NonPos, PSD)
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.reductions import InverseData
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.matrix_stuffing import extract_mip_idx, MatrixStuffing
from cvxpy.reductions.utilities import (are_args_affine,
                                        group_constraints,
                                        lower_equality,
                                        lower_inequality)
from cvxpy.utilities.coeff_extractor import CoeffExtractor


class ParamConeProg(object):
    """Represents a parameterized cone program

    minimize   c'x  + d
    subject to cone_constr1(A_1*x + b_1, ...)
               ...
               cone_constrK(A_i*x + b_i, ...)


    The constant offsets d and b are the last column of c and A.
    """
    def __init__(self, c, x, A, constraints):
        self.c = c
        self.x = x
        self.A = A
        self.contraints = constraints

    def is_mixed_integer(self):
        return self.x.attributes['boolean'] or \
            self.x.attributes['integer']


class ConeMatrixStuffing(MatrixStuffing):
    """Construct matrices for linear cone problems.

    Linear cone problems are assumed to have a linear objective and cone
    constraints which may have zero or more arguments, all of which must be
    affine.
    """

    def accepts(self, problem):
        return (type(problem.objective) == Minimize
                and problem.objective.expr.is_affine()
                and not convex_attributes(problem.variables())
                and are_args_affine(problem.constraints))

    def stuffed_objective(self, problem, extractor):
        # Extract to c.T * x + r
        c = extractor.affine(problem.objective.expr)

        boolean, integer = extract_mip_idx(problem.variables())
        x = Variable(extractor.x_length, boolean=boolean, integer=integer)

        return c, x

    def apply(self, problem):
        inverse_data = InverseData(problem)
        # Form the constraints
        extractor = CoeffExtractor(inverse_data)
        c, x = self.stuffed_objective(problem, extractor)
        # Lower equality and inequality to Zero and NonPos.
        cons = []
        for con in problem.constraints:
            if isinstance(con, Equality):
                con = lower_equality(con)
            elif isinstance(con, Inequality):
                con = lower_inequality(con)
            elif isinstance(con, SOC) and con.axis == 1:
                con = SOC(con.args[0], con.args[1].T, axis=0,
                          constr_id=con.constr_id)
            cons.append(con)
            # TODO this won't work.
            # Need to map from giant constraint map to individual constraints.
            inverse_data.cons_id_map[con.id] = con.id
        # Reorder constraints to Zero, NonPos, SOC, PSD, EXP.
        constr_map = group_constraints()
        ordered_cons = constr_map[Zero] + constr_map[NonPos] + \
            constr_map[SOC] + constr_map[PSD] + constr_map[ExpCone]

        # Batch expressions together, then split apart.
        expr_list = [arg for c in ordered_cons for arg in c.args]
        A = extractor.affine(expr_list)

        # Map of old constraint id to new constraint id.
        inverse_data.minimize = type(problem.objective) == Minimize
        new_prob = ParamConeProg(c, x, A, ordered_cons)
        return new_prob, inverse_data
