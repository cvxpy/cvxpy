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

from cvxpy.cvxcore.python import canonInterface
from cvxpy.constraints import NonPos, Zero, Equality, Inequality
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.reductions import Solution, InverseData
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.matrix_stuffing import extract_mip_idx, MatrixStuffing
from cvxpy.reductions.utilities import (are_args_affine,
                                        lower_equality,
                                        lower_inequality,
                                        group_constraints)
from cvxpy.utilities.coeff_extractor import CoeffExtractor
import cvxpy.settings as s
import numpy as np
import scipy.sparse as sp


class ParamQuadProg(object):
    """Represents a parameterized quadratic program.

    minimize   x'Px  + q^Tx + d
    subject to (in)equality_constr1(A_1*x + b_1, ...)
               ...
               (in)equality_constrK(A_i*x + b_i, ...)


    The constant offsets d and b are the last column of c and A.
    """
    def __init__(self, P, q, x, A,
                 variables,
                 var_id_to_col,
                 constraints,
                 parameters,
                 param_id_to_col,
                 formatted=False):
        self.P = P
        self.q = q
        self.x = x
        self.A = A
        self._A_mapping_nonzero = None
        self.constraints = constraints
        self.constr_size = sum([c.size for c in constraints])
        self.parameters = parameters
        self.param_id_to_col = param_id_to_col
        self.id_to_param = {p.id: p for p in self.parameters}
        self.param_id_to_size = {p.id: p.size for p in self.parameters}
        self.total_param_size = sum([p.size for p in self.parameters])
        # TODO technically part of inverse data.
        self.variables = variables
        self.var_id_to_col = var_id_to_col
        self.id_to_var = {v.id: v for v in self.variables}
        # whether this param cone prog has been formatted for a solver
        self.formatted = formatted

    def is_mixed_integer(self):
        return self.x.attributes['boolean'] or \
            self.x.attributes['integer']

    def apply_parameters(self, id_to_param_value=None, zero_offset=False,
                         keep_zeros=False):
        """Returns A, b after applying parameters (and reshaping).

        Args:
          id_to_param_value: (optional) dict mapping parameter ids to values
          zero_offset: (optional) if True, zero out the constant offset in the
                       parameter vector
          keep_zeros: (optional) if True, store explicit zeros in A where
                        parameters are affected
        """
        def param_value(idx):
            return (np.array(self.id_to_param[idx].value) if id_to_param_value
                    is None else id_to_param_value[idx])
        param_vec = canonInterface.get_parameter_vector(
            self.total_param_size,
            self.param_id_to_col,
            self.param_id_to_size,
            param_value,
            zero_offset=zero_offset)

        # TODO special code to handle P in canonInterface.
        if param_vec is None:
            tensor_application = self.P
        else:
            if sp.issparse(self.P):
                tensor_application = self.P @ sp.csc_matrix(param_vec[:, None])
            else:
                tensor_application = self.P @ param_vec
        P = tensor_application.reshape(
            (self.x.size, self.x.size), order='F').tocsc()

        q, d = canonInterface.get_matrix_and_offset_from_tensor(
            self.q, param_vec, self.x.size)
        q = q.toarray().flatten()
        if keep_zeros and self._A_mapping_nonzero is None:
            self._A_mapping_nonzero = canonInterface.A_mapping_nonzero_rows(
                self.A, self.x.size)
        A, b = canonInterface.get_matrix_and_offset_from_tensor(
            self.A, param_vec, self.x.size,
            nonzero_rows=self._A_mapping_nonzero)
        return P, q, d, A, np.atleast_1d(b)

    def apply_param_jac(self, delP, delq, delA, delb, active_params=None):
        """Multiplies by Jacobian of parameter mapping.

        Assumes delA is sparse.

        Returns:
            A dictionary param.id -> dparam
        """
        raise NotImplementedError

    def split_solution(self, sltn, active_vars=None):
        """Splits the solution into individual variables.
        """
        raise NotImplementedError

    def split_adjoint(self, del_vars=None):
        """Adjoint of split_solution.
        """
        raise NotImplementedError


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
        # extract to 0.5 * x.T * P * x + q.T * x + r
        expr = problem.objective.expr.copy()
        params_to_P, params_to_q = extractor.quad_form(expr)
        # Handle 0.5 factor.
        params_to_P = 2*params_to_P

        # concatenate all variables in one vector
        boolean, integer = extract_mip_idx(problem.variables())
        x = Variable(extractor.x_length, boolean=boolean, integer=integer)

        return params_to_P, params_to_q, x

    def apply(self, problem):
        """See docstring for MatrixStuffing.apply"""
        inverse_data = InverseData(problem)
        # Form the constraints
        extractor = CoeffExtractor(inverse_data)
        params_to_P, params_to_q, flattened_variable = self.stuffed_objective(
            problem, extractor)
        # Lower equality and inequality to Zero and NonPos.
        cons = []
        for con in problem.constraints:
            if isinstance(con, Equality):
                con = lower_equality(con)
            elif isinstance(con, Inequality):
                con = lower_inequality(con)
            cons.append(con)

        # Reorder constraints to Zero, NonPos.
        constr_map = group_constraints(cons)
        ordered_cons = constr_map[Zero] + constr_map[NonPos]
        inverse_data.cons_id_map = {con.id: con.id for con in ordered_cons}

        inverse_data.constraints = ordered_cons
        # Batch expressions together, then split apart.
        expr_list = [arg for c in ordered_cons for arg in c.args]
        params_to_Ab = extractor.affine(expr_list)

        inverse_data.minimize = type(problem.objective) == Minimize
        new_prob = ParamQuadProg(params_to_P,
                                 params_to_q,
                                 flattened_variable,
                                 params_to_Ab,
                                 problem.variables(),
                                 inverse_data.var_offsets,
                                 ordered_cons,
                                 problem.parameters(),
                                 inverse_data.param_id_map)
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

        return Solution(solution.status, opt_val, primal_vars, dual_vars,
                        solution.attr)
