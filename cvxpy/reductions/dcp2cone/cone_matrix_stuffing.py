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
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import (
    PSD,
    SOC,
    Equality,
    ExpCone,
    Inequality,
    NonNeg,
    NonPos,
    PowCone3D,
    Zero,
)
from cvxpy.cvxcore.python import canonInterface
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.param_prob import ParamProb
from cvxpy.reductions import InverseData, Solution, cvx_attr2constr
from cvxpy.reductions.matrix_stuffing import MatrixStuffing, extract_mip_idx
from cvxpy.reductions.utilities import (
    ReducedMat,
    are_args_affine,
    group_constraints,
    lower_equality,
    lower_ineq_to_nonneg,
    nonpos2nonneg,
)
from cvxpy.utilities.coeff_extractor import CoeffExtractor


class ConeDims:
    """Summary of cone dimensions present in constraints.

    Constraints must be formatted as dictionary that maps from
    constraint type to a list of constraints of that type.

    Attributes
    ----------
    zero : int
        The dimension of the zero cone.
    nonneg : int
        The dimension of the non-negative cone.
    exp : int
        The number of 3-dimensional exponential cones.
    soc : list of int
        A list of the second-order cone dimensions.
    psd : list of int
        A list of the positive semidefinite cone dimensions, where the
        dimension of the PSD cone of k by k matrices is k.
    """

    EQ_DIM = s.EQ_DIM
    LEQ_DIM = s.LEQ_DIM
    EXP_DIM = s.EXP_DIM
    SOC_DIM = s.SOC_DIM
    PSD_DIM = s.PSD_DIM
    P3D_DIM = 'p3'

    def __init__(self, constr_map) -> None:
        self.zero = int(sum(c.size for c in constr_map[Zero]))
        self.nonneg = int(sum(c.size for c in constr_map[NonNeg]))
        self.exp = int(sum(c.num_cones() for c in constr_map[ExpCone]))
        self.soc = [int(dim) for c in constr_map[SOC] for dim in c.cone_sizes()]
        self.psd = [int(c.shape[0]) for c in constr_map[PSD]]
        p3d = []
        if constr_map[PowCone3D]:
            p3d = np.concatenate([c.alpha.value for c in constr_map[PowCone3D]]).tolist()
        self.p3d = p3d

    def __repr__(self) -> str:
        return "(zero: {0}, nonneg: {1}, exp: {2}, soc: {3}, psd: {4}, p3d: {5})".format(
            self.zero, self.nonneg, self.exp, self.soc, self.psd, self.p3d)

    def __str__(self) -> str:
        """String representation.
        """
        return ("%i equalities, %i inequalities, %i exponential cones, \n"
                "SOC constraints: %s, PSD constraints: %s,\n"
                " 3d power cones %s.") % (self.zero,
                                          self.nonneg,
                                          self.exp,
                                          self.soc,
                                          self.psd,
                                          self.p3d)

    def __getitem__(self, key):
        if key == self.EQ_DIM:
            return self.zero
        elif key == self.LEQ_DIM:
            return self.nonneg
        elif key == self.EXP_DIM:
            return self.exp
        elif key == self.SOC_DIM:
            return self.soc
        elif key == self.PSD_DIM:
            return self.psd
        elif key == self.P3D_DIM:
            return self.p3d
        else:
            raise KeyError(key)


# TODO(akshayka): unit tests
class ParamConeProg(ParamProb):
    """Represents a parameterized cone program

    minimize   c'x  + d + [(1/2)x'Px]
    subject to cone_constr1(A_1*x + b_1, ...)
               ...
               cone_constrK(A_i*x + b_i, ...)


    The constant offsets d and b are the last column of c and A.
    """
    def __init__(self, c, x, A,
                 variables,
                 var_id_to_col,
                 constraints,
                 parameters,
                 param_id_to_col,
                 P=None,
                 formatted: bool = False) -> None:
        # The problem data tensors; c is for the constraint, and A for
        # the problem data matrix
        self.c = c
        self.A = A
        self.P = P
        # The variable
        self.x = x

        # Form a reduced representation of A and P, for faster application
        # of parameters.
        self.reduced_A = ReducedMat(self.A, self.x.size)
        self.reduced_P = ReducedMat(self.P, self.x.size, quad_form=True)

        self.constraints = constraints
        self.constr_size = sum([c.size for c in constraints])
        self.constr_map = group_constraints(constraints)
        self.cone_dims = ConeDims(self.constr_map)
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

    def is_mixed_integer(self) -> bool:
        """Is the problem mixed-integer?"""
        return self.x.attributes['boolean'] or \
            self.x.attributes['integer']

    def apply_parameters(self, id_to_param_value=None, zero_offset: bool = False,
                         keep_zeros: bool = False, quad_obj: bool = False):
        """Returns A, b after applying parameters (and reshaping).

        Args:
          id_to_param_value: (optional) dict mapping parameter ids to values.
          zero_offset: (optional) if True, zero out the constant offset in the
                       parameter vector.
          keep_zeros: (optional) if True, store explicit zeros in A where
                        parameters are affected.
          quad_obj: (optional) if True, include quadratic objective term.
        """
        self.reduced_A.cache(keep_zeros)

        def param_value(idx):
            return (np.array(self.id_to_param[idx].value) if id_to_param_value
                    is None else id_to_param_value[idx])

        param_vec = canonInterface.get_parameter_vector(
            self.total_param_size,
            self.param_id_to_col,
            self.param_id_to_size,
            param_value,
            zero_offset=zero_offset)
        c, d = canonInterface.get_matrix_from_tensor(
            self.c, param_vec, self.x.size, with_offset=True)
        c = c.toarray().flatten()
        A, b = self.reduced_A.get_matrix_from_tensor(param_vec, with_offset=True)
        if quad_obj:
            self.reduced_P.cache(keep_zeros)
            P, _ = self.reduced_P.get_matrix_from_tensor(param_vec, with_offset=False)
            return P, c, d, A, np.atleast_1d(b)
        else:
            return c, d, A, np.atleast_1d(b)

    def apply_param_jac(self, delc, delA, delb, active_params=None):
        """Multiplies by Jacobian of parameter mapping.

        Assumes delA is sparse.

        Returns:
            A dictionary param.id -> dparam
        """
        if self.P is not None:
            raise ValueError("Can't apply Jacobian with a quadratic objective.")

        if active_params is None:
            active_params = {p.id for p in self.parameters}

        del_param_vec = delc @ self.c[:-1]
        flatdelA = delA.reshape((np.prod(delA.shape), 1), order='F')
        delAb = sp.vstack([flatdelA, sp.csc_matrix(delb[:, None])])

        one_gig_of_doubles = 125000000
        if delAb.shape[0] < one_gig_of_doubles:
            # fast path: if delAb is small enough, just materialize it
            # in memory because sparse-matrix @ dense vector is much faster
            # than sparse @ sparse
            del_param_vec += np.squeeze(self.A.T.dot(delAb.toarray()))
        else:
            # slow path.
            # TODO: make this faster by intelligently operating on the
            # sparse matrix data / making use of reduced_A
            del_param_vec += np.squeeze((delAb.T @ self.A).A)
        del_param_vec = np.squeeze(del_param_vec)

        param_id_to_delta_param = {}
        for param_id, col in self.param_id_to_col.items():
            if param_id in active_params:
                param = self.id_to_param[param_id]
                delta = del_param_vec[col:col + param.size]
                param_id_to_delta_param[param_id] = np.reshape(
                    delta, param.shape, order='F')
        return param_id_to_delta_param

    def split_solution(self, sltn, active_vars=None):
        """Splits the solution into individual variables.
        """
        if active_vars is None:
            active_vars = [v.id for v in self.variables]
        # var id to solution.
        sltn_dict = {}
        for var_id, col in self.var_id_to_col.items():
            if var_id in active_vars:
                var = self.id_to_var[var_id]
                value = sltn[col:var.size+col]
                if var.attributes_were_lowered():
                    orig_var = var.variable_of_provenance()
                    value = cvx_attr2constr.recover_value_for_variable(
                        orig_var, value, project=False)
                    sltn_dict[orig_var.id] = np.reshape(
                        value, orig_var.shape, order='F')
                else:
                    sltn_dict[var_id] = np.reshape(
                        value, var.shape, order='F')
        return sltn_dict

    def split_adjoint(self, del_vars=None):
        """Adjoint of split_solution.
        """
        var_vec = np.zeros(self.x.size)
        for var_id, delta in del_vars.items():
            var = self.id_to_var[var_id]
            col = self.var_id_to_col[var_id]
            if var.attributes_were_lowered():
                orig_var = var.variable_of_provenance()
                if cvx_attr2constr.attributes_present(
                        [orig_var], cvx_attr2constr.SYMMETRIC_ATTRIBUTES):
                    delta = delta + delta.T - np.diag(np.diag(delta))
                delta = cvx_attr2constr.lower_value(orig_var, delta)
            var_vec[col:col + var.size] = delta.flatten(order='F')
        return var_vec


class ConeMatrixStuffing(MatrixStuffing):
    """Construct matrices for linear cone problems.

    Linear cone problems are assumed to have a linear objective and cone
    constraints which may have zero or more arguments, all of which must be
    affine.
    """
    CONSTRAINTS = 'ordered_constraints'

    def __init__(self, quad_obj: bool = False, canon_backend: str | None = None):
        # Assume a quadratic objective?
        self.quad_obj = quad_obj
        self.canon_backend = canon_backend

    def accepts(self, problem):
        valid_obj_curv = (self.quad_obj and problem.objective.expr.is_quadratic()) or \
            problem.objective.expr.is_affine()
        return (type(problem.objective) == Minimize
                and valid_obj_curv
                and not cvx_attr2constr.convex_attributes(problem.variables())
                and are_args_affine(problem.constraints)
                and problem.is_dpp())

    def stuffed_objective(self, problem, extractor):
        # concatenate all variables in one vector
        boolean, integer = extract_mip_idx(problem.variables())
        x = Variable(extractor.x_length, boolean=boolean, integer=integer)
        if self.quad_obj:
            # extract to 0.5 * x.T * P * x + q.T * x + r
            expr = problem.objective.expr.copy()
            params_to_P, params_to_c = extractor.quad_form(expr)
            # Handle 0.5 factor.
            params_to_P = 2*params_to_P
        else:
            # Extract to c.T * x + r; c is represented by a ma
            params_to_c = extractor.affine(problem.objective.expr)
            params_to_P = None
        return params_to_P, params_to_c, x

    def apply(self, problem):
        inverse_data = InverseData(problem)
        # Form the constraints
        extractor = CoeffExtractor(inverse_data, self.canon_backend)
        params_to_P, params_to_c, flattened_variable = self.stuffed_objective(
            problem, extractor)
        # Lower equality and inequality to Zero and NonNeg.
        cons = []
        for con in problem.constraints:
            if isinstance(con, Equality):
                con = lower_equality(con)
            elif isinstance(con, Inequality):
                con = lower_ineq_to_nonneg(con)
            elif isinstance(con, NonPos):
                con = nonpos2nonneg(con)
            elif isinstance(con, SOC) and con.axis == 1:
                con = SOC(con.args[0], con.args[1].T, axis=0,
                          constr_id=con.constr_id)
            elif isinstance(con, PowCone3D) and con.args[0].ndim > 1:
                x, y, z = con.args
                alpha = con.alpha
                con = PowCone3D(x.flatten(), y.flatten(), z.flatten(), alpha.flatten(),
                                constr_id=con.constr_id)
            elif isinstance(con, ExpCone) and con.args[0].ndim > 1:
                x, y, z = con.args
                con = ExpCone(x.flatten(), y.flatten(), z.flatten(),
                              constr_id=con.constr_id)
            cons.append(con)
        # Reorder constraints to Zero, NonNeg, SOC, PSD, EXP, PowCone3D
        constr_map = group_constraints(cons)
        ordered_cons = constr_map[Zero] + constr_map[NonNeg] + \
            constr_map[SOC] + constr_map[PSD] + constr_map[ExpCone] + constr_map[PowCone3D]
        inverse_data.cons_id_map = {con.id: con.id for con in ordered_cons}

        inverse_data.constraints = ordered_cons
        # Batch expressions together, then split apart.
        expr_list = [arg for c in ordered_cons for arg in c.args]
        params_to_problem_data = extractor.affine(expr_list)

        inverse_data.minimize = type(problem.objective) == Minimize
        new_prob = ParamConeProg(params_to_c,
                                 flattened_variable,
                                 params_to_problem_data,
                                 problem.variables(),
                                 inverse_data.var_offsets,
                                 ordered_cons,
                                 problem.parameters(),
                                 inverse_data.param_id_map,
                                 P=params_to_P)
        return new_prob, inverse_data

    def invert(self, solution, inverse_data):
        """Retrieves a solution to the original problem"""
        var_map = inverse_data.var_offsets
        con_map = inverse_data.cons_id_map
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
            for old_con, new_con in con_map.items():
                con_obj = inverse_data.id2cons[old_con]
                shape = con_obj.shape
                # TODO rationalize Exponential.
                if shape == () or isinstance(con_obj, (ExpCone, SOC)):
                    dual_vars[old_con] = solution.dual_vars[new_con]
                else:
                    dual_vars[old_con] = np.reshape(
                        solution.dual_vars[new_con],
                        shape,
                        order='F'
                    )

        return Solution(solution.status, opt_val, primal_vars, dual_vars,
                        solution.attr)
