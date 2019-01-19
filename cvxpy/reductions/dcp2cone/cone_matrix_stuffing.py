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

from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.matrix_stuffing import extract_mip_idx, MatrixStuffing
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.utilities import are_args_affine


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
        return (type(problem.objective) == Minimize
                and problem.objective.expr.is_affine()
                and not convex_attributes(problem.variables())
                and are_args_affine(problem.constraints))

    def stuffed_objective(self, problem, extractor):
        # Extract to c.T * x + r
        C, R = extractor.affine(problem.objective.expr)

        c = C.toarray().flatten()
        boolean, integer = extract_mip_idx(problem.variables())
        x = Variable(extractor.N, boolean=boolean, integer=integer)

        new_obj = c.T * x + 0

        return new_obj, x, R[0]
