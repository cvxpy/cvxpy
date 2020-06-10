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

import cvxpy.settings as s
from cvxpy.constraints import SOC, ExpCone, PSD, Zero, NonNeg
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.reductions.solvers import utilities
import numpy as np
import scipy.sparse as sp


# NOTE(akshayka): Small changes to this file can lead to drastic
# performance regressions. If you are making a change to this file,
# make sure to run cvxpy/tests/test_benchmarks.py to ensure that you have
# not introduced a regression.

class LinearOperator(object):
    """A wrapper for linear operators."""
    def __init__(self, linear_op, shape):
        if sp.issparse(linear_op):
            self._matmul = lambda X: linear_op @ X
        else:
            self._matmul = linear_op
        self.shape = shape

    def __call__(self, X):
        return self._matmul(X)


def as_linear_operator(linear_op):
    if isinstance(linear_op, LinearOperator):
        return linear_op
    elif sp.issparse(linear_op):
        return LinearOperator(linear_op, linear_op.shape)


def as_block_diag_linear_operator(matrices):
    """Block diag of SciPy sparse matrices or linear operators."""
    linear_operators = [as_linear_operator(op) for op in matrices]
    nrows = [op.shape[0] for op in linear_operators]
    ncols = [op.shape[1] for op in linear_operators]
    m, n = sum(nrows), sum(ncols)
    col_indices = np.append(0, np.cumsum(ncols))

    def matmul(X):
        outputs = []
        for i, op in enumerate(linear_operators):
            Xi = X[col_indices[i]:col_indices[i + 1]]
            outputs.append(op(Xi))
        return sp.vstack(outputs)
    return LinearOperator(matmul, (m, n))


class ConicSolver(Solver):
    """Conic solver class with reduction semantics
    """
    # The key that maps to ConeDims in the data returned by apply().
    DIMS = "dims"

    # Every conic solver must support Zero and NonNeg constraints.
    SUPPORTED_CONSTRAINTS = [Zero, NonNeg]

    # Some solvers cannot solve problems that do not have constraints.
    # For such solvers, REQUIRES_CONSTR should be set to True.
    REQUIRES_CONSTR = False

    EXP_CONE_ORDER = None

    def accepts(self, problem):
        return (isinstance(problem, ParamConeProg)
                and (self.MIP_CAPABLE or not problem.is_mixed_integer())
                and not convex_attributes([problem.x])
                and (len(problem.constraints) > 0 or not self.REQUIRES_CONSTR)
                and all(type(c) in self.SUPPORTED_CONSTRAINTS for c in
                        problem.constraints))

    @staticmethod
    def get_spacing_matrix(shape, spacing, streak, num_blocks, offset):
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
        return sp.csc_matrix((val_arr, (row_arr, col_arr)), shape)

    def psd_format_mat(self, constr):
        """Return a matrix to multiply by PSD constraint coefficients.
        """
        # Default is identity.
        return sp.eye(constr.size, format='csc')

    def format_constraints(self, problem, exp_cone_order):
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
        # Create a matrix to reshape constraints, then replicate for each
        # variable entry.
        restruct_mat = []  # Form a block diagonal matrix.
        for constr in problem.constraints:
            total_height = sum([arg.size for arg in constr.args])
            if type(constr) == Zero:
                restruct_mat.append(-sp.eye(constr.size, format='csr'))
            elif type(constr) == NonNeg:
                restruct_mat.append(sp.eye(constr.size, format='csr'))
            elif type(constr) == SOC:
                # Group each t row with appropriate X rows.
                assert constr.axis == 0, 'SOC must be lowered to axis == 0'

                # Interleave the rows of coeffs[0] and coeffs[1]:
                #     coeffs[0][0, :]
                #     coeffs[1][0:gap-1, :]
                #     coeffs[0][1, :]
                #     coeffs[1][gap-1:2*(gap-1), :]
                t_spacer = ConicSolver.get_spacing_matrix(
                    shape=(total_height, constr.args[0].size),
                    spacing=constr.args[1].shape[0],
                    streak=1,
                    num_blocks=constr.args[0].size,
                    offset=0,
                )
                X_spacer = ConicSolver.get_spacing_matrix(
                    shape=(total_height, constr.args[1].size),
                    spacing=1,
                    streak=constr.args[1].shape[0],
                    num_blocks=constr.args[0].size,
                    offset=1,
                )
                restruct_mat.append(sp.hstack([t_spacer, X_spacer]))
            elif type(constr) == ExpCone:
                arg_mats = []
                for i, arg in enumerate(constr.args):
                    space_mat = ConicSolver.get_spacing_matrix(
                        shape=(total_height, arg.size),
                        spacing=len(exp_cone_order) - 1,
                        streak=1,
                        num_blocks=arg.size,
                        offset=exp_cone_order[i],
                    )
                    arg_mats.append(space_mat)
                restruct_mat.append(sp.hstack(arg_mats))
            elif type(constr) == PSD:
                restruct_mat.append(self.psd_format_mat(constr))
            else:
                raise ValueError("Unsupported constraint type.")

        # Form new ParamConeProg
        if restruct_mat:
            # TODO(akshayka): profile to see whether using linear operators
            # or bmat is faster
            restruct_mat = as_block_diag_linear_operator(restruct_mat)
            # this is equivalent to but _much_ faster than:
            #    restruct_mat_rep = sp.block_diag([restruct_mat]*(problem.x.size + 1))
            #    restruct_A = restruct_mat_rep * problem.A
            unspecified, remainder = divmod(problem.A.shape[0] *
                                            problem.A.shape[1],
                                            restruct_mat.shape[1])
            reshaped_A = problem.A.reshape(restruct_mat.shape[1],
                                           unspecified, order='F').tocsr()
            restructured_A = restruct_mat(reshaped_A).tocoo()
            # Because of a bug in scipy versions <  1.20, `reshape`
            # can overflow if indices are int32s.
            restructured_A.row = restructured_A.row.astype(np.int64)
            restructured_A.col = restructured_A.col.astype(np.int64)
            restructured_A = restructured_A.reshape(
                restruct_mat.shape[0] * (problem.x.size + 1),
                problem.A.shape[1], order='F')
        else:
            restructured_A = problem.A
        new_param_cone_prog = ParamConeProg(problem.c,
                                            problem.x,
                                            restructured_A,
                                            problem.variables,
                                            problem.var_id_to_col,
                                            problem.constraints,
                                            problem.parameters,
                                            problem.param_id_to_col,
                                            formatted=True)
        return new_param_cone_prog

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']

        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value']
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
