"""
Copyright 2017 Robin Verschueren, 2017 Akshay Agrawal

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
from cvxpy.constraints import PSD, SOC, ExpCone, NonNeg, PowCone3D, PowConeND, SvecPSD, Zero
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.solver import Solver

# NOTE(akshayka): Small changes to this file can lead to drastic
# performance regressions. If you are making a change to this file,
# make sure to run cvxpy/tests/test_benchmarks.py to ensure that you have
# not introduced a regression.


def dims_to_solver_dict(cone_dims):
    cones = {
        'f': cone_dims.zero,
        'l': cone_dims.nonneg,
        'q': cone_dims.soc,
        'ep': cone_dims.exp,
        's': cone_dims.psd,
        'p': cone_dims.p3d,
        'pnd': cone_dims.pnd
    }
    return cones


def _spacing_rows(spacing, streak, num_blocks, offset):
    """Destination rows of :meth:`ConicSolver.get_spacing_matrix`'s columns.

    Column ``k`` of that matrix has its single nonzero in row
    ``result[k]``; this is the same index expression, without the matrix.
    """
    return (np.arange(num_blocks * (streak + spacing))
            .reshape(num_blocks, streak + spacing)[:, :streak].ravel() + offset)


def _reformatted(problem, restructured_A):
    """``problem`` with its constraint rows in the solver's cone layout."""
    return ParamConeProg(
        problem.q,
        problem.x,
        restructured_A,
        problem.variables,
        problem.var_id_to_col,
        problem.constraints,
        problem.parameters,
        problem.param_id_to_col,
        P=problem.P,
        formatted=True,
        lower_bounds=problem.lower_bounds,
        upper_bounds=problem.upper_bounds,
        lb_tensor=problem.lb_tensor,
        ub_tensor=problem.ub_tensor,
        dir_cones=problem.dir_cones,
    )


def _identity_rows(constr, exp_cone_order):
    """Zero, NonNeg, PSD and SvecPSD keep every row where it is."""
    return np.arange(constr.size)


def _soc_rows(constr, exp_cone_order):
    """Group each t row with the X rows of its own cone."""
    assert constr.axis == 0, 'SOC must be lowered to axis == 0'
    # Handle scalar X (shape is empty tuple)
    x_dim = constr.args[1].shape[0] if constr.args[1].shape else 1
    num_cones = constr.args[0].size
    return np.concatenate([_spacing_rows(x_dim, 1, num_cones, 0),
                           _spacing_rows(1, x_dim, num_cones, 1)])


def _exp_rows(constr, exp_cone_order):
    """Interleave (x, y, z) per cone, in the solver's argument order."""
    return np.concatenate([
        _spacing_rows(len(exp_cone_order) - 1, 1, arg.size, exp_cone_order[i])
        for i, arg in enumerate(constr.args)])


def _pow3d_rows(constr, exp_cone_order):
    """Interleave (x, y, z) per cone."""
    return np.concatenate([_spacing_rows(2, 1, arg.size, i)
                           for i, arg in enumerate(constr.args)])


def _pownd_rows(constr, exp_cone_order):
    """Each cone is a column of W followed by its entry of z."""
    W = constr.args[0]
    m, n = (W.shape[0], 1) if W.ndim == 1 else W.shape
    assert constr.args[1].size == n
    return np.concatenate([_spacing_rows(0, 1, m, (m + 1) * j) for j in range(n)]
                          + [_spacing_rows(m, 1, n, m)])


# Where each constraint type's rows land within its own block.
_ROW_LAYOUT = {
    Zero: _identity_rows,
    NonNeg: _identity_rows,
    PSD: _identity_rows,
    SvecPSD: _identity_rows,
    SOC: _soc_rows,
    ExpCone: _exp_rows,
    PowCone3D: _pow3d_rows,
    PowConeND: _pownd_rows,
}


def restruct_permutation(constraints, exp_cone_order):
    """Where each stuffed constraint row moves, and its sign.

    The cone-restructuring matrix R is always a signed permutation: every
    block is either +-I (Zero contributes -I; NonNeg, PSD and SvecPSD
    contribute I) or a pure interleaving of the constraint's arguments
    (SOC and the exponential/power cones). So R never has to be built, let
    alone multiplied by -- restructuring is a row remap plus a sign flip.

    Purely structural: derived from the constraint types and shapes plus
    ``exp_cone_order``, never from coefficient or parameter values.

    Returns ``(new_row, sign)``, both indexed by the *old* row: row ``i``
    becomes row ``new_row[i]``, scaled by ``sign[i]``. ``None`` when there are
    no constraints, i.e. nothing to reorder.
    """
    if not constraints:
        return None
    new_row, sign, base = [], [], 0
    for constr in constraints:
        layout = _ROW_LAYOUT.get(type(constr))
        if layout is None:
            raise ValueError("Unsupported constraint type.")
        rows = layout(constr, exp_cone_order)
        new_row.append(rows + base)
        # Only the zero cone is negated; it enters as -I.
        sign.append(np.full(rows.size, -1.0 if type(constr) == Zero else 1.0))
        base += rows.size
    return np.concatenate(new_row), np.concatenate(sign)


def _permute_rows(A, new_row, sign):
    """Apply a signed row permutation to a stuffed parameter tensor.

    The tensor stacks the ``x.size + 1`` columns of ``[A | b]`` vertically, so
    tensor row ``i + m*j`` holds concrete row ``i`` of column ``j``: scattering
    row ``i`` to ``new_row[i]`` shifts every one of its stored values by the
    same amount. Returns ``A`` untouched when there is nothing to apply.
    """
    m = new_row.shape[0]
    shift = new_row - np.arange(m)
    moves = shift.any()
    if not moves and (sign > 0).all():
        return A
    A = A.tocsc(copy=True)
    row = A.indices % m
    if moves:
        # Shifting, rather than recomputing new_row[i] + m*j, keeps the
        # arithmetic in the index dtype, which scipy requires to match indptr.
        A.indices = A.indices + shift.astype(A.indices.dtype)[row]
        # Shifting rows leaves each column's indices out of ascending order.
        A.sort_indices()
    A.data = A.data * sign[row]
    return A


class ConicSolver(Solver):
    """Conic solver class with reduction semantics

    ``apply`` requires a program whose constraint rows are already in this
    solver's cone layout, i.e. ``problem.formatted`` is True. The
    ``ConeFormat`` reduction establishes that: ``_build_solving_chain``
    appends it before every non-QP solver. Interfaces must not re-derive it.
    """
    # The key that maps to ConeDims in the data returned by apply().
    DIMS = "dims"

    # Every conic solver must support Zero and NonNeg constraints.
    SUPPORTED_CONSTRAINTS = [Zero, NonNeg]

    # Some solvers cannot solve problems that do not have constraints.
    # For such solvers, REQUIRES_CONSTR should be set to True.
    REQUIRES_CONSTR = False

    # If a solver supports exponential cones, it must specify the corresponding order
    # The cvxpy standard for the exponential cone is:
    #     K_e = closure{(x,y,z) |  z >= y * exp(x/y), y>0}.
    # Whenever a solver uses this convention, EXP_CONE_ORDER should be [0, 1, 2].
    EXP_CONE_ORDER = None

    def accepts(self, problem):
        return (isinstance(problem, ParamConeProg)
                and (self.MIP_CAPABLE or not problem.is_mixed_integer())
                and not convex_attributes([problem.x])
                and (len(problem.constraints) > 0 or not self.REQUIRES_CONSTR)
                and all(type(c) in self.SUPPORTED_CONSTRAINTS for c in
                        problem.constraints))

    @staticmethod
    def get_spacing_matrix(shape: tuple[int, ...], spacing, streak, num_blocks, offset):
        """Returns a sparse matrix that spaces out an expression.

        Parameters
        ----------
        shape : tuple
            (rows in matrix, columns in matrix)
        spacing : int
            The number of rows between the start of each non-zero block.
        streak: int
            The number of elements in each block.
        num_blocks : int
            The number of non-zero blocks.
        offset : int
            The number of zero rows at the beginning of the matrix.

        Returns
        -------
        SciPy CSC matrix
            A sparse matrix
        """
        num_values = num_blocks * streak
        val_arr = np.ones(num_values, dtype=np.float64)
        streak_plus_spacing = streak + spacing
        row_arr = np.arange(0, num_blocks * streak_plus_spacing).reshape(
            num_blocks, streak_plus_spacing)[:, :streak].flatten() + offset
        col_arr = np.arange(num_values)
        return sp.csc_array((val_arr, (row_arr, col_arr)), shape)

    @classmethod
    def format_constraints(cls, problem, exp_cone_order):
        """
        Returns a ParamConeProg whose problem data tensors will yield the
        coefficient "A" and offset "b" for the constraint in the following
        formats:
            Linear equations: (A, b) such that A * x + b == 0,
            Linear inequalities: (A, b) such that A * x + b >= 0,
            Second order cone: (A, b) such that A * x + b in SOC,
            Exponential cone: (A, b) such that A * x + b in EXP,
            Semidefinite cone: (A, b) such that A * x + b in PSD,

        The CVXPY standard for the exponential cone is:
            K_e = closure{(x,y,z) |  z >= y * exp(x/y), y>0}.
        Whenever a solver uses this convention, EXP_CONE_ORDER should be
        [0, 1, 2].

        The CVXPY standard for the second order cone is:
            SOC(n) = { x : x[0] >= norm(x[1:n], 2)  }.
        All currently supported solvers use this convention.

        Args:
          problem : ParamConeProg
            The problem that is the provenance of the constraint.
          exp_cone_order: list
            A list indicating how the exponential cone arguments are ordered.

        Returns:
          ParamConeProg with structured A.
        """
        # R is a signed permutation, so applying it is a row shift and a sign
        # flip -- no reshape, no block-diagonal matmul.
        perm = restruct_permutation(problem.constraints, exp_cone_order)
        A = problem.A if perm is None else _permute_rows(problem.A, *perm)
        return _reformatted(problem, A)

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']

        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value'] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: solution['primal']}
            eq_dual = utilities.get_dual_values(
                solution['eq_dual'],
                utilities.extract_dual_value,
                inverse_data[Solver.EQ_CONSTR])
            leq_dual = utilities.get_dual_values(
                solution['ineq_dual'],
                utilities.extract_dual_value,
                inverse_data[Solver.NEQ_CONSTR])
            eq_dual.update(leq_dual)
            dual_vars = eq_dual
            return Solution(status, opt_val, primal_vars, dual_vars, {})
        else:
            return failure_solution(status)

    def _prepare_data_and_inv_data(self, problem):
        data = {}
        inv_data = {self.VAR_ID: problem.x.id}

        # Rows arrive in this order, established by the ConeFormat reduction.
        # By default cvxpy follows the SCS convention, which requires
        # constraints to be specified in the following order:
        # 1. zero cone
        # 2. non-negative orthant
        # 3. soc
        # 4. psd
        # 5. exponential
        # 6. three-dimensional power cones
        # 7. n-dimensional power cones
        data[s.PARAM_PROB] = problem
        data[self.DIMS] = problem.cone_dims
        inv_data[self.DIMS] = problem.cone_dims

        constr_map = problem.constr_map
        inv_data[self.EQ_CONSTR] = constr_map[Zero]
        inv_data[self.NEQ_CONSTR] = constr_map[NonNeg] + constr_map[SOC] + \
            constr_map.get(PSD, []) + constr_map.get(SvecPSD, []) + \
            constr_map[ExpCone] + \
            constr_map[PowCone3D] + \
            constr_map[PowConeND]
        return problem, data, inv_data

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """

        # This is a reference implementation following SCS conventions
        # Implementations for other solvers may amend or override the implementation entirely

        problem, data, inv_data = self._prepare_data_and_inv_data(problem)

        # Apply parameter values.
        # Obtain A, b such that Ax + s = b, s \in cones.
        if not problem.has_quad_obj:
            c, d, A, b = problem.apply_parameters()
        else:
            P, c, d, A, b = problem.apply_parameters(quad_obj=True)
            data[s.P] = P
        data[s.C] = c
        inv_data[s.OFFSET] = d
        data[s.A] = -A
        data[s.B] = b
        data[s.LOWER_BOUNDS] = problem.lower_bounds
        data[s.UPPER_BOUNDS] = problem.upper_bounds
        return data, inv_data
