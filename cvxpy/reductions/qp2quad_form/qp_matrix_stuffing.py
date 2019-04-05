"""
Copyright 2016 Jaehyun Park, 2017 Robin Verschueren

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

from cvxpy.atoms import QuadForm
from cvxpy.constraints import NonPos, Zero, Equality, Inequality
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.matrix_stuffing import extract_mip_idx, MatrixStuffing
from cvxpy.reductions.utilities import are_args_affine


class QpMatrixStuffing(MatrixStuffing):
    """Fills in numeric values for this problem instance.

       Outputs a DCP-compliant minimization problem with an objective
       of the form
           QuadForm(x, p) + q.T * x
       and Zero/NonPos constraints, both of which exclusively carry
       affine arguments.
    """

    @staticmethod
    def accepts(problem):
        return (type(problem.objective) == Minimize
                and problem.objective.is_quadratic()
                and problem.is_dcp()
                and not convex_attributes(problem.variables())
                and all(type(c) in [Zero, NonPos, Equality, Inequality]
                        for c in problem.constraints)
                and are_args_affine(problem.constraints))

    def stuffed_objective(self, problem, extractor):
        # extract to x.T * P * x + q.T * x + r
        expr = problem.objective.expr.copy()
        P, q, r = extractor.quad_form(expr)

        # concatenate all variables in one vector
        boolean, integer = extract_mip_idx(problem.variables())
        x = Variable(extractor.N, boolean=boolean, integer=integer)
        new_obj = QuadForm(x, P) + q.T*x

        return new_obj, x, r
