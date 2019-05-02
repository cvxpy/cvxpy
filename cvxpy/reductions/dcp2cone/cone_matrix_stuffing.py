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

import cvxpy.settings as s
import cvxpy.lin_ops.lin_op as lo
from cvxpy.constraints import (Equality, ExpCone, Inequality,
                               SOC, Zero, NonPos, PSD)
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.reductions import InverseData, Solution
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.matrix_stuffing import extract_mip_idx, MatrixStuffing
from cvxpy.reductions.utilities import (are_args_affine,
                                        group_constraints,
                                        lower_equality,
                                        lower_inequality)
from cvxpy.utilities.coeff_extractor import CoeffExtractor
import numpy as np
import scipy.sparse as sp


class ParamConeProg(object):
    """Represents a parameterized cone program

    minimize   c'x  + d
    subject to cone_constr1(A_1*x + b_1, ...)
               ...
               cone_constrK(A_i*x + b_i, ...)


    The constant offsets d and b are the last column of c and A.
    """
    def __init__(self, c, x, A,
                 constraints,
                 parameters,
                 param_id_to_col):
        self.c = c
        self.x = x
        self.A = A
        self.constraints = constraints
        self.constr_size = sum([c.size for c in constraints])
        self.parameters = parameters
        self.param_id_to_col = param_id_to_col
        self.id_to_param = {p.id: p for p in self.parameters}
        self.total_param_size = sum([p.size for p in self.parameters])

    def is_mixed_integer(self):
        return self.x.attributes['boolean'] or \
            self.x.attributes['integer']

    def apply_parameters(self):
        """Returns A, b after applying parameters (and reshaping).
        """
        # Flatten parameters.
        param_vec = np.zeros(self.total_param_size + 1)
        # TODO handle parameters with structure.
        for param_id, col in self.param_id_to_col.items():
            if param_id == lo.CONSTANT_ID:
                param_vec[col] = 1
            else:
                param = self.id_to_param[param_id]
                value = np.array(param.value).flatten(order='F')
                param_vec[col:param.size+col] = value
        # New problem without parameters.
        c = (self.c*param_vec).flatten()
        # Need to cast to sparse matrix.
        param_vec = sp.csc_matrix(param_vec[:, None])
        A = (self.A*param_vec).reshape((self.constr_size, self.x.size + 1), order='F')
        A = A.tocsc()
        new_prob = ParamConeProg(c, self.x, A,
                                 self.constraints, [], {})
        return new_prob


class ConeMatrixStuffing(MatrixStuffing):
    """Construct matrices for linear cone problems.

    Linear cone problems are assumed to have a linear objective and cone
    constraints which may have zero or more arguments, all of which must be
    affine.
    """
    CONSTRAINTS = 'ordered_constraints'

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
        # Reorder constraints to Zero, NonPos, SOC, PSD, EXP.
        constr_map = group_constraints(cons)
        ordered_cons = constr_map[Zero] + constr_map[NonPos] + \
            constr_map[SOC] + constr_map[PSD] + constr_map[ExpCone]

        inverse_data.constraints = ordered_cons
        # Batch expressions together, then split apart.
        expr_list = [arg for c in ordered_cons for arg in c.args]
        A = extractor.affine(expr_list)

        # Map of old constraint id to new constraint id.
        inverse_data.minimize = type(problem.objective) == Minimize
        new_prob = ParamConeProg(c, x, A, ordered_cons,
                                 problem.parameters(),
                                 inverse_data.param_id_map)
        return new_prob, inverse_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data."""
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
                dual_vars[constr.id] = []
                for arg in constr.args:
                    dual_vars[constr.id].append(
                        np.reshape(
                            dual_var[offset:offset+arg.size],
                            arg.shape,
                            order='F'
                        )
                    )
                    offset += arg.size

        return Solution(solution.status, opt_val, primal_vars, dual_vars,
                        solution.attr)
