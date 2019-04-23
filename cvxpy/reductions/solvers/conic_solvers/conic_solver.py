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
from cvxpy.constraints import SOC, ExpCone, NonPos, PSD, Zero
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.reductions.solvers import utilities
import numpy as np
import scipy.sparse as sp


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
    # The key for the parameterized problem.
    PARAM_PROB = "param_prob"

    # Every conic solver must support Zero and NonPos constraints.
    SUPPORTED_CONSTRAINTS = [Zero, NonPos]

    # Some solvers cannot solve problems that do not have constraints.
    # For such solvers, REQUIRES_CONSTR should be set to True.
    REQUIRES_CONSTR = False

    def accepts(self, problem):
        return (type(problem.objective) == Minimize
                and (self.MIP_CAPABLE or not problem.is_mixed_integer())
                and not convex_attributes(problem.x)
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
        val_arr = []
        row_arr = []
        col_arr = []
        # Selects from each column.
        for var_row in range(num_blocks):
            for i in range(streak):
                val_arr.append(np.float64(1.0))
                row_arr.append((streak + spacing)*var_row + i + offset)
                col_arr.append(var_row*streak + i)
        return sp.csc_matrix((val_arr, (row_arr, col_arr)), shape)

    def format_constraints(self, problem, exp_cone_order):
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
          problem : ParamConeProg
            The problem that is the provenance of the constraint.
          exp_cone_order: list
            A list indicating how the exponential cone arguments are ordered.

        Returns:
          ParamConeProg with structured A.
        """
        # Create a matrix to reshape constraints,
        # then replicate for each variable entry.
        restruct_mat = []  # Form a block diagonal matrix.
        for constr in problem.constraints:
            total_height = sum([arg.size for arg in constr.args])
            if type(constr) in [Zero, NonPos]:
                # Both of these constraints have but a single argument.
                # c.T * x + b (<)= 0 if and only if c.T * x (<)= -b.
                # Need to negate to switch from NonPos to NonNeg.
                restruct_mat.append(-sp.eye(constr.size, format='csc'))
            elif type(constr) == SOC:
                # Group each t row with appropriate X rows.
                assert constr.axis == 0, 'SOC must be lowered to axis == 0'

                # Interleave the rows of coeffs[0] and coeffs[1]:
                #     coeffs[0][0, :]
                #     coeffs[1][0:gap-1, :]
                #     coeffs[0][1, :]
                #     coeffs[1][gap-1:2*(gap-1), :]
                t_spacer = ConicSolver.get_spacing_matrix(
                    (total_height, constr.args[0].size),
                    constr.args[1].shape[0],
                    1,
                    constr.args[0].size,
                    0,
                )
                X_spacer = ConicSolver.get_spacing_matrix(
                    (total_height, constr.args[1].size),
                    1,
                    constr.args[1].shape[0],
                    constr.args[0].size,
                    1,
                )
                restruct_mat.append(sp.hstack([t_spacer, X_spacer]))
            elif type(constr) == ExpCone:
                for i, arg in enumerate(constr.args):
                    space_mat = ConicSolver.get_spacing_matrix(
                        (total_height, arg.size),
                        len(exp_cone_order),
                        1,
                        exp_cone_order[i])
                    restruct_mat.append(space_mat)
            elif type(constr) == PSD:
                # Sign flipped relative to NonPos, Zero.
                # TODO -A, b
                restruct_mat.append(sp.eye(constr.size, format='csc'))
            else:
                raise ValueError("Unsupported constraint type.")

        # Form new ParamConeProg
        restruct_mat = sp.block_diag(restruct_mat)
        restruct_mat_rep = sp.block_diag([restruct_mat]*(problem.x.size + 1))
        restruct_A = restruct_mat_rep*problem.A
        new_param_cone_prog = ParamConeProg(problem.c,
                                            problem.x,
                                            restruct_A,
                                            problem.constraints,
                                            problem.parameters,
                                            problem.param_id_to_col)
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
