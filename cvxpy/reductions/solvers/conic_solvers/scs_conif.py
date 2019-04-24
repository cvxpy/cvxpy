"""
Copyright 2013 Steven Diamond, 2017 Robin Verschueren, 2017 Akshay Agrawal

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
from cvxpy.constraints import PSD, SOC, ExpCone, NonPos, Zero
from cvxpy.expressions.constants.constant import Constant
import cvxpy.interface as intf
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.solution import failure_solution, Solution
from cvxpy.reductions.solvers.solver import group_constraints
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (ConeDims,
                                                                 ConicSolver)
from cvxpy.reductions.solvers import utilities
from cvxpy.utilities.coeff_extractor import CoeffExtractor


# Utility method for formatting a ConeDims instance into a dictionary
# that can be supplied to scs.
def dims_to_solver_dict(cone_dims):
    cones = {
        "f": cone_dims.zero,
        "l": cone_dims.nonpos,
        "q": cone_dims.soc,
        "ep": cone_dims.exp,
        "s": cone_dims.psd,
    }
    return cones


# Utility methods for special handling of semidefinite constraints.
def scaled_lower_tri(matrix):
    """Returns an expression representing the lower triangular entries

    Scales the strictly lower triangular entries by sqrt(2), as required
    by SCS.

    Parameters
    ----------
    matrix : Expression
        A 2-dimensional CVXPY expression.

    Returns
    -------
    Expression
        An expression representing the (scaled) lower triangular part of
        the supplied matrix expression.
    """
    rows = cols = matrix.shape[0]
    entries = rows * (cols + 1)//2

    row_arr = np.arange(0, entries)

    lower_diag_indices = np.tril_indices(rows)
    col_arr = np.sort(np.ravel_multi_index(lower_diag_indices, (rows, cols), order='F'))

    val_arr = np.zeros((rows, cols))
    val_arr[lower_diag_indices] = np.sqrt(2)
    np.fill_diagonal(val_arr, 1)
    val_arr = np.ravel(val_arr, order='F')
    val_arr = val_arr[np.nonzero(val_arr)]

    shape = (entries, rows*cols)
    coeff = Constant(sp.csc_matrix((val_arr, (row_arr, col_arr)), shape))
    vectorized_matrix = reshape(matrix, (rows*cols, 1))
    return coeff * vectorized_matrix


def tri_to_full(lower_tri, n):
    """Expands n*(n+1)//2 lower triangular to full matrix

    Scales off-diagonal by 1/sqrt(2), as per the SCS specification.

    Parameters
    ----------
    lower_tri : numpy.ndarray
        A NumPy array representing the lower triangular part of the
        matrix, stacked in column-major order.
    n : int
        The number of rows (columns) in the full square matrix.

    Returns
    -------
    numpy.ndarray
        A 2-dimensional ndarray that is the scaled expansion of the lower
        triangular array.
    """
    full = np.zeros((n, n))
    full[np.triu_indices(n)] = lower_tri
    full[np.tril_indices(n)] = lower_tri

    full[np.tril_indices(n, k=-1)] /= np.sqrt(2)
    full[np.triu_indices(n, k=1)] /= np.sqrt(2)

    return np.reshape(full, n*n, order="F")


class SCS(ConicSolver):
    """An interface for the SCS solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC,
                                                                 ExpCone,
                                                                 PSD]
    REQUIRES_CONSTR = True

    # Map of SCS status to CVXPY status.
    STATUS_MAP = {"Solved": s.OPTIMAL,
                  "Solved/Inaccurate": s.OPTIMAL_INACCURATE,
                  "Unbounded": s.UNBOUNDED,
                  "Unbounded/Inaccurate": s.UNBOUNDED_INACCURATE,
                  "Infeasible": s.INFEASIBLE,
                  "Infeasible/Inaccurate": s.INFEASIBLE_INACCURATE,
                  "Failure": s.SOLVER_ERROR,
                  "Indeterminate": s.SOLVER_ERROR,
                  "Interrupted": s.SOLVER_ERROR}

    # Order of exponential cone arguments for solver.
    EXP_CONE_ORDER = [0, 1, 2]

    def name(self):
        """The name of the solver.
        """
        return s.SCS

    def import_solver(self):
        """Imports the solver.
        """
        import scs
        scs  # For flake8

    def format_constr(self, problem, constr, exp_cone_order):
        """Extract coefficient and offset vector from constraint.

        Special cases PSD constraints, as SCS expects constraints to be
        imposed on solely the lower triangular part of the variable matrix.
        Moreover, it requires the off-diagonal coefficients to be scaled by
        sqrt(2).
        """
        if isinstance(constr, PSD):
            expr = constr.expr
            triangularized_expr = scaled_lower_tri(expr + expr.T)/2
            extractor = CoeffExtractor(InverseData(problem))
            A_prime, b_prime = extractor.affine(triangularized_expr)
            # SCS requests constraints to be formatted as
            # Ax + s = b, where s is constrained to reside in some
            # cone. Here, however, we are formatting the constraint
            # as A"x + b" = s = -Ax + b; hence, A = -A", b = b"
            return -1 * A_prime, b_prime
        else:
            return super(SCS, self).format_constr(problem, constr,
                                                  exp_cone_order)

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data = {}
        inv_data = {self.VAR_ID: problem.variables()[0].id}

        # Parse the coefficient vector from the objective.
        data[s.C], data[s.OFFSET] = self.get_coeff_offset(
            problem.objective.args[0])
        data[s.C] = data[s.C].ravel()
        inv_data[s.OFFSET] = data[s.OFFSET][0]

        # Order and group nonlinear constraints.
        constr_map = group_constraints(problem.constraints)
        data[ConicSolver.DIMS] = ConeDims(constr_map)
        inv_data[ConicSolver.DIMS] = data[ConicSolver.DIMS]

        # SCS requires constraints to be specified in the following order:
        # 1. zero cone
        # 2. non-negative orthant
        # 3. soc
        # 4. psd
        # 5. exponential
        zero_constr = constr_map[Zero]
        neq_constr = (constr_map[NonPos] + constr_map[SOC]
                      + constr_map[PSD] + constr_map[ExpCone])
        inv_data[SCS.EQ_CONSTR] = zero_constr
        inv_data[SCS.NEQ_CONSTR] = neq_constr

        # Obtain A, b such that Ax + s = b, s \in cones.
        #
        # Note that scs mandates that the cones MUST be ordered with
        # zero cones first, then non-nonnegative orthant, then SOC,
        # then PSD, then exponential.
        data[s.A], data[s.B] = self.group_coeff_offset(
            problem, zero_constr + neq_constr, self.EXP_CONE_ORDER)
        return data, inv_data

    def extract_dual_value(self, result_vec, offset, constraint):
        """Extracts the dual value for constraint starting at offset.

        Special cases PSD constraints, as per the SCS specification.
        """
        if isinstance(constraint, PSD):
            dim = constraint.shape[0]
            lower_tri_dim = dim * (dim + 1) // 2
            new_offset = offset + lower_tri_dim
            lower_tri = result_vec[offset:new_offset]
            full = tri_to_full(lower_tri, dim)
            return full, new_offset
        else:
            return utilities.extract_dual_value(result_vec, offset,
                                                constraint)

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = self.STATUS_MAP[solution["info"]["status"]]

        attr = {}
        attr[s.SOLVE_TIME] = solution["info"]["solveTime"]
        attr[s.SETUP_TIME] = solution["info"]["setupTime"]
        attr[s.NUM_ITERS] = solution["info"]["iter"]

        if status in s.SOLUTION_PRESENT:
            primal_val = solution["info"]["pobj"]
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {
                inverse_data[SCS.VAR_ID]:
                intf.DEFAULT_INTF.const_to_matrix(solution["x"])
            }
            eq_dual_vars = utilities.get_dual_values(
                intf.DEFAULT_INTF.const_to_matrix(
                    solution["y"][:inverse_data[ConicSolver.DIMS].zero]),
                self.extract_dual_value,
                inverse_data[SCS.EQ_CONSTR])
            ineq_dual_vars = utilities.get_dual_values(
                intf.DEFAULT_INTF.const_to_matrix(
                    solution["y"][inverse_data[ConicSolver.DIMS].zero:]),
                self.extract_dual_value,
                inverse_data[SCS.NEQ_CONSTR])
            dual_vars = {}
            dual_vars.update(eq_dual_vars)
            dual_vars.update(ineq_dual_vars)
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status)

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        warm_start : Bool
            Whether to warm_start SCS.
        verbose : Bool
            Control the verbosity.
        solver_opts : dict
            SCS-specific solver options.

        Returns
        -------
        The result returned by a call to scs.solve().
        """
        import scs
        args = {"A": data[s.A], "b": data[s.B], "c": data[s.C]}
        if warm_start and solver_cache is not None and \
           self.name() in solver_cache:
            args["x"] = solver_cache[self.name()]["x"]
            args["y"] = solver_cache[self.name()]["y"]
            args["s"] = solver_cache[self.name()]["s"]
        cones = dims_to_solver_dict(data[ConicSolver.DIMS])
        # Default to eps = 1e-4 instead of 1e-3.
        solver_opts['eps'] = solver_opts.get('eps', 1e-4)
        results = scs.solve(
            args,
            cones,
            verbose=verbose,
            **solver_opts)
        if solver_cache is not None:
            solver_cache[self.name()] = results
        return results
