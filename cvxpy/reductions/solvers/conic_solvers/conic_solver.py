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

from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression, multiply
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.constraints import SOC, ExpCone, NonPos, PSD, Zero
from cvxpy.expressions.constants.constant import Constant
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
import cvxpy.settings as s


def is_stuffed_cone_constraint(constraint):
    """Conic solvers require constraints to be stuffed in the following way.
    """
    if len(constraint.variables()) != 1:
        return False
    for arg in constraint.args:
        if type(arg) == reshape:
            arg = arg.args[0]
        if type(arg) == AddExpression:
            if type(arg.args[0]) not in [MulExpression, multiply]:
                return False
            if not arg.args[0].args[0].is_constant():
                return False
            if not arg.args[1].is_constant():
                return False
        elif type(arg) in [MulExpression, multiply]:
            if not arg.args[0].is_constant():
                return False
        else:
            return False
    return True


def is_stuffed_cone_objective(objective):
    """Conic solvers require objectives to be stuffed in the following way.
    """
    expr = objective.expr
    return (expr.is_affine()
            and len(expr.variables()) == 1
            and type(expr) == AddExpression
            and len(expr.args) == 2
            and type(expr.args[0]) in [MulExpression, multiply]
            and type(expr.args[1]) == Constant)


class ConeDims(object):
    """Summary of cone dimensions present in constraints.

    Constraints must be formatted as dictionary that maps from
    constraint type to a list of constraints of that type.

    Attributes
    ----------
    zero : int
        The dimension of the zero cone.
    nonpos : int
        The dimension of the non-positive cone.
    exp : int
        The dimension of the exponential cone.
    soc : list of int
        A list of the second-order cone dimensions.
    psd : list of int
        A list of the positive semidefinite cone dimensions, where the
        dimension of the PSD cone of k by k matrices is k.
    """
    def __init__(self, constr_map):
        self.zero = int(sum(c.size for c in constr_map[Zero]))
        self.nonpos = int(sum(c.size for c in constr_map[NonPos]))
        self.exp = int(sum(c.num_cones() for c in constr_map[ExpCone]))
        self.soc = [int(dim) for c in constr_map[SOC] for dim in c.cone_sizes()]
        self.psd = [int(c.shape[0]) for c in constr_map[PSD]]

    def __repr__(self):
        return "(zero: {0}, nonpos: {1}, exp: {2}, soc: {3}, psd: {4})".format(
            self.zero, self.nonpos, self.exp, self.soc, self.psd)


class ConicSolver(Solver):
    """Conic solver class with reduction semantics
    """
    # The key that maps to ConeDims in the data returned by apply().
    DIMS = "dims"

    # Every conic solver must support Zero and NonPos constraints.
    SUPPORTED_CONSTRAINTS = [Zero, NonPos]

    # Some solvers cannot solve problems that do not have constraints.
    # For such solvers, REQUIRES_CONSTR should be set to True.
    REQUIRES_CONSTR = False

    def accepts(self, problem):
        return (type(problem.objective) == Minimize
                and (self.MIP_CAPABLE or not problem.is_mixed_integer())
                and is_stuffed_cone_objective(problem.objective)
                and not convex_attributes(problem.variables())
                and (len(problem.constraints) > 0 or not self.REQUIRES_CONSTR)
                and all(type(c) in self.SUPPORTED_CONSTRAINTS for c in
                        problem.constraints)
                and all(is_stuffed_cone_constraint(c) for c in
                        problem.constraints))

    @staticmethod
    def get_coeff_offset(expr):
        """Return the coefficient A and offset b in A*x + b.

        Args:
          expr: A CVXPY expression.

        Returns:
          (SciPy COO sparse matrix, NumPy 1D array)
        """
        # May be a reshape as root.
        if type(expr) == reshape:
            expr = expr.args[0]
        # Convert data to float64.
        if len(expr.args[0].args) == 0:
            # expr is c.T*x
            offset = 0
            coeff = expr.args[0].value.astype(np.float64)
        else:
            # expr is c.T*x + d
            offset = expr.args[1].value.ravel().astype(np.float64)
            coeff = expr.args[0].args[0].value.astype(np.float64)
        # Convert scalars to sparse matrices.
        if np.isscalar(coeff):
            coeff = sp.coo_matrix(([coeff], ([0], [0])), shape=(1, 1))
        return (coeff, offset)

    @staticmethod
    def get_spacing_matrix(shape, spacing, offset):
        """Returns a sparse matrix that spaces out an expression.

        Parameters
        ----------
        shape : tuple
            (rows in matrix, columns in matrix)
        spacing : int
            The number of rows between each non-zero.
        offset : int
            The number of zero rows at the beginning of the matrix.

        Returns
        -------
        SciPy CSC matrix
            A sparse matrix
        """
        val_arr = []
        row_arr = []
        col_arr = []
        # Selects from each column.
        for var_row in range(shape[1]):
            val_arr.append(np.float64(1.0))
            row_arr.append(spacing*var_row + offset)
            col_arr.append(var_row)
        return sp.csc_matrix((val_arr, (row_arr, col_arr)), shape)

    def format_constr(self, problem, constr, exp_cone_order):
        """
        Return the coefficient "A" and offset "b" for the constraint in the
        following formats:
            Linear equations: (A, b) such that A * x == b,
            Linear inequalities: (A, b) such that A * x <= b,
            Second order cone: (A, b) such that A * x <=_{SOC} b,
            Exponential cone: (A, b) such that A * x <=_{EXP} b,
            Semidefinite cone: (A, b) such that A * x <=_{SDP} b,

        The CVXPY standard for the exponential cone is:
            K_e = closure{(x,y,z) |  y >= z * exp(x/z), z>0}.
        Whenever a solver uses this convention, EXP_CONE_ORDER should be
        [0, 1, 2].

        The CVXPY standard for the second order cone is:
            SOC(n) = { x : x[0] >= norm(x[1:n], 2)  }.
        All currently supported solvers use this convention.

        Args:
          problem : Problem
            The problem that is the provenance of the constraint.
          constr : Constraint.
            The constraint to format.

        Returns:
          (SciPy CSC sparse matrix, NumPy 1D array)
        """
        coeffs, offsets = [], []
        for arg in constr.args:
            coeff, offset = ConicSolver.get_coeff_offset(arg)
            coeffs.append(coeff)
            offsets.append(offset)
        height = sum(c.shape[0] for c in coeffs)

        if type(constr) in [NonPos, Zero]:
            # Both of these constraints have a single argument.
            # c.T * x + b (<)= 0 if and only if c.T * x (<)= -b.
            return coeffs[0].tocsc(), -offsets[0]
        elif type(constr) == SOC:
            # Group each t row with appropriate X rows.
            assert constr.axis == 0, "SOC must be lowered to axis == 0"

            # coeffs[0] corresponds to the scalar part `t`, coeffs[1] to `X`
            #
            # Interleave the rows of coeffs[0] and coeffs[1]:
            #     coeffs[0][0, :]
            #     coeffs[1][0:gap-1, :]
            #     coeffs[0][1, :]
            #     coeffs[1][gap-1:2*(gap-1), :]
            #     <etc.>
            # where `gap` == constr.args[1].shape[0], i.e., the number of
            # rows in `X` The vectorized code below implements this
            # interleaving.
            X_coeff = coeffs[1].tocoo()
            # Because of a bug in scipy versions <= 1.20, `reshape`
            # occasionally overflows if indices are int32s.
            #
            # This might cause issues on windows, due to an overflow bug in
            # `reshape`
            X_coeff.row = X_coeff.row.astype(np.int64)
            X_coeff.col = X_coeff.col.astype(np.int64)
            reshaped = X_coeff.reshape((coeffs[0].shape[0], -1))
            stacked = -sp.hstack([coeffs[0], reshaped])
            stacked.row = stacked.row.astype(np.int64)
            stacked.col = stacked.col.astype(np.int64)
            stacked = stacked.reshape(
              (coeffs[0].shape[0] + X_coeff.shape[0], coeffs[0].shape[1]))

            offset = np.hstack([
              np.expand_dims(offsets[0], 1),
              offsets[1].reshape((offsets[0].shape[0], -1))]).ravel()
            return stacked.tocsc(), offset
        elif type(constr) == ExpCone:
            for i, coeff in enumerate(coeffs):
                mat = ConicSolver.get_spacing_matrix(
                                                (height, coeff.shape[0]),
                                                len(exp_cone_order),
                                                exp_cone_order[i])
                offsets[i] = mat*offsets[i]
                coeffs[i] = -mat*coeffs[i]
            return sum(coeffs).tocsc(), sum(offsets)
        elif type(constr) == PSD:
            # Sign flipped relative to NonPos, Zero.
            return -coeffs[0].tocsc(), offsets[0]
        else:
            # subclasses must handle PSD constraints.
            raise ValueError("Unsupported constraint type.")

    def group_coeff_offset(self, problem, constraints, exp_cone_order):
        """Combine the constraints into a single matrix A, offset b.

        Parameters
        ----------
          problem: Problem
            The CVXPY problem that is the provenance of the constraints.
          constraints: list of Constraint
            The constraints to process.
        Returns
        -------
          (SciPy CSC sparse matrix, NumPy 1D array)
        """
        if not constraints:
            return None, None
        matrices, offsets = [], []
        for cons in constraints:
            coeff, offset = self.format_constr(problem, cons, exp_cone_order)
            matrices.append(coeff)
            offsets.append(offset)
        coeff = sp.vstack(matrices).tocsc()
        offset = np.hstack(offsets)
        return coeff, offset

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
