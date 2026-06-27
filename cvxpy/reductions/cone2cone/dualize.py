"""
Copyright, the CVXPY authors

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

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.constraints.nonpos import Inequality, NonNeg
from cvxpy.constraints.zero import Equality, Zero
from cvxpy.cvxcore.python import canonInterface
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.utilities import lower_equality, lower_ineq_to_nonneg
from cvxpy.utilities.coeff_extractor import CoeffExtractor


class Dualize(Reduction):
    """Transforms a conic minimization problem into its Lagrangian dual.

    This reduction operates at the expression level (before ConeMatrixStuffing).
    It takes a problem of the form:

        (P)  min  c'x + d
             s.t. A_i x + b_i in K_i,   i = 1, ..., m

    and produces the dual problem:

        (D)  max  -sum(b_i' y_i) + d
             s.t. sum(A_i' y_i) = c
                  y_i in K_i*,           i = 1, ..., m

    which is then reformulated as a minimization problem and returned as a
    standard CVXPY Problem that can be processed by the rest of the chain.

    The dual cone K_i* is constructed using each constraint's ``_dual_cone``
    method. Primal variable values are recovered from the dual of the
    equality constraint, and original constraint duals are recovered from
    the dual problem's primal variable values.

    Assumptions
    -----------
    - The problem is a minimization with affine objective.
    - All constraint arguments are affine.
    - No integer or boolean variables (strong duality requires continuity).
    - No parameters (DPP not yet supported).

    Position in the chain
    ---------------------
    This reduction should be placed after Dcp2Cone + CvxAttr2Constr
    (and optional Cone2Cone reductions) but before ConeMatrixStuffing::

        ... -> Dcp2Cone -> CvxAttr2Constr -> [Cone2Cone] -> Dualize
            -> ConeMatrixStuffing -> Solver
    """

    def accepts(self, problem) -> bool:
        if type(problem.objective) != Minimize:
            return False
        if not problem.objective.expr.is_affine():
            return False
        if problem.parameters():
            return False
        for v in problem.variables():
            if v.attributes.get('boolean') or v.attributes.get('integer'):
                return False
        for c in problem.constraints:
            for arg in c.args:
                if not arg.is_affine():
                    return False
        return True

    def apply(self, problem):
        constraints = problem.constraints
        if not constraints:
            return problem, None

        # Lower Equality -> Zero and Inequality -> NonNeg (same as
        # ConeMatrixStuffing) so we only need to handle primitive cones.
        lowered = []
        cons_id_map = {}  # maps original con id -> lowered con id
        for con in constraints:
            if isinstance(con, Equality):
                lc = lower_equality(con)
            elif isinstance(con, Inequality):
                lc = lower_ineq_to_nonneg(con)
            else:
                lc = con
            lowered.append(lc)
            cons_id_map[con.id] = lc.id
        constraints = lowered

        inv = InverseData(problem)
        n = inv.x_length

        # Extract coefficient matrices (non-parametric path).
        extractor = CoeffExtractor(inv, 'SCIPY')

        # Objective: c'x + d
        obj_tensor = extractor.affine(problem.objective.expr)
        c_mat, d_arr = canonInterface.get_matrix_from_tensor(
            obj_tensor, None, n, with_offset=True)
        c_vec = np.asarray(c_mat.todense()).flatten()
        d_val = float(np.asarray(d_arr).flatten()[0])

        # Constraints: stack all args and extract [A | b].
        expr_list = [arg for con in constraints for arg in con.args]
        constr_tensor = extractor.affine(expr_list)
        A_mat, b_arr = canonInterface.get_matrix_from_tensor(
            constr_tensor, None, n, with_offset=True)
        b_vec = np.asarray(b_arr).flatten()

        # --- Build dual problem ---
        m = A_mat.shape[0]
        y = Variable(m, name='dual_y')

        # Dual objective (as minimization):
        #   min  b'y - d   (negation of the dual max{-b'y + d})
        dual_obj = Constant(b_vec) @ y + Constant(-d_val)

        # Dual equality: A' y = c  →  Zero(A'y - c)
        A_T = sp.csc_matrix(A_mat.T)
        eq_expr = Constant(A_T) @ y - Constant(c_vec)
        eq_constr = Zero(eq_expr)
        dual_constraints = [eq_constr]

        # Dual cone constraints for each primal constraint.
        constr_offsets = {}
        offset = 0
        for con in constraints:
            arg_sizes = [arg.size for arg in con.args]
            total_size = sum(arg_sizes)
            constr_offsets[con.id] = (offset, total_size)

            if isinstance(con, Zero):
                # Dual of the zero cone is R^n (free): no constraint.
                pass
            elif isinstance(con, NonNeg):
                # R_+ is self-dual.
                dual_constraints.append(NonNeg(y[offset:offset + total_size]))
            else:
                # Use the constraint's _dual_cone method with reshaped
                # slices of y matching the original arg shapes.
                dual_args = []
                arg_off = offset
                for arg in con.args:
                    if arg.size == 1 and arg.shape == ():
                        # Scalar arg: index to get scalar expression.
                        y_arg = y[arg_off]
                    else:
                        y_slice = y[arg_off:arg_off + arg.size]
                        if arg.shape != y_slice.shape:
                            y_arg = reshape(y_slice, arg.shape, order='F')
                        else:
                            y_arg = y_slice
                    dual_args.append(y_arg)
                    arg_off += arg.size
                dual_constraints.append(con._dual_cone(*dual_args))

            offset += total_size

        dual_problem = Problem(Minimize(dual_obj), dual_constraints)

        # Map constr_offsets to use original (pre-lowering) constraint IDs.
        orig_constr_offsets = {}
        for orig_con in problem.constraints:
            lowered_id = cons_id_map[orig_con.id]
            orig_constr_offsets[orig_con.id] = constr_offsets[lowered_id]

        inverse_data = {
            'var_offsets': inv.var_offsets,
            'var_shapes': inv.var_shapes,
            'eq_constr_id': eq_constr.id,
            'y_id': y.id,
            'constr_offsets': orig_constr_offsets,
        }
        return dual_problem, inverse_data

    def invert(self, solution, inverse_data):
        if inverse_data is None:
            return solution

        # Swap infeasible <-> unbounded (duality).
        _STATUS_MAP = {
            s.INFEASIBLE: s.UNBOUNDED,
            s.UNBOUNDED: s.INFEASIBLE,
            s.INFEASIBLE_INACCURATE: s.UNBOUNDED_INACCURATE,
            s.UNBOUNDED_INACCURATE: s.INFEASIBLE_INACCURATE,
        }
        status = _STATUS_MAP.get(solution.status, solution.status)

        # The dual-as-minimize has opt_val = -p*; negate to recover p*.
        opt_val = -solution.opt_val if solution.opt_val is not None else None

        primal_vars = {}
        dual_vars = {}

        if status in s.SOLUTION_PRESENT:
            # Primal variable values come from the equality constraint's dual.
            eq_dual = solution.dual_vars.get(inverse_data['eq_constr_id'])
            if eq_dual is not None:
                eq_dual = np.asarray(eq_dual).flatten()
                for var_id, var_offset in inverse_data['var_offsets'].items():
                    shape = inverse_data['var_shapes'][var_id]
                    size = int(np.prod(shape))
                    primal_vars[var_id] = np.reshape(
                        eq_dual[var_offset:var_offset + size], shape, order='F')

            # Original constraint duals come from y's optimal value.
            y_val = solution.primal_vars.get(inverse_data['y_id'])
            if y_val is not None:
                y_val = np.asarray(y_val).flatten()
                for con_id, (off, sz) in inverse_data['constr_offsets'].items():
                    dual_vars[con_id] = y_val[off:off + sz]

        return Solution(status, opt_val, primal_vars, dual_vars, solution.attr)
