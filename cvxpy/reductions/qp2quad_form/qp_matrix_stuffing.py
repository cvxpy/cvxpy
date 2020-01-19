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

from cvxpy import problems
from cvxpy.atoms import QuadForm, reshape
from cvxpy.constraints import NonPos, Zero, Equality, Inequality
from cvxpy.cvxcore.python import canonInterface as canon
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.reductions import Solution, InverseData
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.matrix_stuffing import extract_mip_idx, MatrixStuffing
from cvxpy.reductions.utilities import are_args_affine, lower_equality, lower_inequality
import cvxpy.settings as s
from cvxpy.utilities.coeff_extractor import CoeffExtractor

import numpy as np


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
                and are_args_affine(problem.constraints)
                and problem.is_dpp())

    def stuffed_objective(self, problem, extractor):
        # extract to x.T * P * x + q.T * x + r
        expr = problem.objective.expr.copy()
        P, q, r = extractor.quad_form(expr)

        # concatenate all variables in one vector
        boolean, integer = extract_mip_idx(problem.variables())
        x = Variable(extractor.x_length, boolean=boolean, integer=integer)
        new_obj = QuadForm(x, P) + q.T @ x

        return new_obj, x, r

    def apply(self, problem):
        """See docstring for MatrixStuffing.apply"""
        inverse_data = InverseData(problem)
        # Form the constraints
        extractor = CoeffExtractor(inverse_data)
        new_obj, new_var, r = self.stuffed_objective(problem, extractor)
        inverse_data.r = r
        # Lower equality and inequality to Zero and NonPos.
        cons = []
        for con in problem.constraints:
            if isinstance(con, Equality):
                con = lower_equality(con)
            elif isinstance(con, Inequality):
                con = lower_inequality(con)
            cons.append(con)

        # Batch expressions together, then split apart.
        expr_list = [arg for c in cons for arg in c.args]
        problem_data_tensor = extractor.affine(expr_list)
        Afull, bfull = canon.get_matrix_and_offset_from_unparameterized_tensor(
            problem_data_tensor, new_var.size)
        if 0 not in Afull.shape and 0 not in bfull.shape:
            Afull = cvxtypes.constant()(Afull)
            bfull = cvxtypes.constant()(np.atleast_1d(bfull))

        new_cons = []
        offset = 0
        for orig_con, con in zip(problem.constraints, cons):
            arg_list = []
            for arg in con.args:
                A = Afull[offset:offset+arg.size, :]
                b = bfull[offset:offset+arg.size]
                arg_list.append(reshape(A @ new_var + b, arg.shape))
                offset += arg.size
            new_constraint = con.copy(arg_list)
            new_cons.append(new_constraint)

        inverse_data.constraints = new_cons
        inverse_data.minimize = type(problem.objective) == Minimize
        new_prob = problems.problem.Problem(Minimize(new_obj), new_cons)
        return new_prob, inverse_data

    def invert(self, solution, inverse_data):
        """Retrieves the solution to the original problem."""
        var_map = inverse_data.var_offsets
        # Flip sign of opt val if maximize.
        opt_val = solution.opt_val
        if solution.status not in s.ERROR and not inverse_data.minimize:
            opt_val = -solution.opt_val

        primal_vars, dual_vars = {}, {}
        if solution.status not in s.SOLUTION_PRESENT:
            return Solution(solution.status, opt_val, primal_vars, dual_vars,
                            solution.attr)

        # Split vectorized variable into components.
        x_opt = list(solution.primal_vars.values())[0]
        for var_id, offset in var_map.items():
            shape = inverse_data.var_shapes[var_id]
            size = np.prod(shape, dtype=int)
            primal_vars[var_id] = np.reshape(x_opt[offset:offset+size], shape,
                                             order='F')

        # Remap dual variables if dual exists (problem is convex).
        if solution.dual_vars is not None:
            # Giant dual variable.
            dual_var = list(solution.dual_vars.values())[0]
            offset = 0
            for constr in inverse_data.constraints:
                # QP constraints can only have one argument.
                dual_vars[constr.id] = np.reshape(
                    dual_var[offset:offset+constr.args[0].size],
                    constr.args[0].shape,
                    order='F'
                )
                offset += constr.size

        # Add constant part
        if inverse_data.minimize:
            opt_val += inverse_data.r
        else:
            opt_val -= inverse_data.r
        return Solution(solution.status, opt_val, primal_vars, dual_vars,
                        solution.attr)
