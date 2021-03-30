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

import cvxpy.settings as s
from cvxpy.constraints import Zero, NonNeg, PSD, SOC, ExpCone, PowCone3D
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.solution import failure_solution, Solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers import utilities
import numpy as np
import scipy.sparse as sp


# Utility method for formatting a ConeDims instance into a dictionary
# that can be supplied to scs.
def dims_to_solver_dict(cone_dims):
    cones = {
        'f': cone_dims.zero,
        'l': cone_dims.nonneg,
        'q': cone_dims.soc,
        'ep': cone_dims.exp,
        's': cone_dims.psd,
        'p': cone_dims.p3d
    }
    return cones


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

    Notes
    -----
    SCS tracks "lower triangular" indices in a way that corresponds to numpy's
    "upper triangular" indices. So the function call below uses ``np.triu_indices``
    in a way that looks weird, but is nevertheless correct.
    """
    full = np.zeros((n, n))
    full[np.triu_indices(n)] = lower_tri
    full += full.T
    full[np.diag_indices(n)] /= 2
    full[np.tril_indices(n, k=-1)] /= np.sqrt(2)
    full[np.triu_indices(n, k=1)] /= np.sqrt(2)
    return np.reshape(full, n*n, order="F")


def scs_psdvec_to_psdmat(vec: Expression, indices: np.ndarray) -> Expression:
    """
    Return "V" so that "vec[indices] belongs to the SCS-standard PSD cone"
    can be written in natural cvxpy syntax as "V >> 0".

    Parameters
    ----------
    vec : cvxpy.expressions.expression.Expression
        Must have ``vec.is_affine() == True``.
    indices : ndarray
        Contains nonnegative integers, which can index into ``vec``.

    Notes
    -----
    This function is similar to ``tri_to_full``, which is also found
    in this file. The difference is that this function works without
    indexed assignment ``mat[i,j] = expr``. Such indexed assignment
    cannot be used, because this function builds a cvxpy Expression,
    rather than a numpy ndarray.
    """
    n = int(np.sqrt(indices.size * 2))
    rows, cols = np.triu_indices(n)
    mats = []
    for i, idx in enumerate(indices):
        r, c = rows[i], cols[i]
        mat = np.zeros(shape=(n, n))
        if r == c:
            mat[r, r] = 1
        else:
            mat[r, c] = 1 / np.sqrt(2)
            mat[c, r] = 1 / np.sqrt(2)
        mat = vec[idx] * mat
        mats.append(mat)
    V = sum(mats)
    return V


class SCS(ConicSolver):
    """An interface for the SCS solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS \
        + [SOC, ExpCone, PSD, PowCone3D]
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

    def import_solver(self) -> None:
        """Imports the solver.
        """
        import scs
        scs  # For flake8

    def psd_format_mat(self, constr):
        """Return a linear operator to multiply by PSD constraint coefficients.

        Special cases PSD constraints, as SCS expects constraints to be
        imposed on solely the lower triangular part of the variable matrix.
        Moreover, it requires the off-diagonal coefficients to be scaled by
        sqrt(2), and applies to the symmetric part of the constrained expression.
        """
        rows = cols = constr.expr.shape[0]
        entries = rows * (cols + 1)//2

        row_arr = np.arange(0, entries)

        lower_diag_indices = np.tril_indices(rows)
        col_arr = np.sort(np.ravel_multi_index(lower_diag_indices,
                                               (rows, cols),
                                               order='F'))

        val_arr = np.zeros((rows, cols))
        val_arr[lower_diag_indices] = np.sqrt(2)
        np.fill_diagonal(val_arr, 1.0)
        val_arr = np.ravel(val_arr, order='F')
        val_arr = val_arr[np.nonzero(val_arr)]

        shape = (entries, rows*cols)
        scaled_lower_tri = sp.csc_matrix((val_arr, (row_arr, col_arr)), shape)

        idx = np.arange(rows * cols)
        val_symm = 0.5 * np.ones(2 * rows * cols)
        K = idx.reshape((rows, cols))
        row_symm = np.append(idx, np.ravel(K, order='F'))
        col_symm = np.append(idx, np.ravel(K.T, order='F'))
        symm_matrix = sp.csc_matrix((val_symm, (row_symm, col_symm)))

        return scaled_lower_tri @ symm_matrix

    def _prepare_data_and_inv_data(self, problem):
        data = {}
        inv_data = {self.VAR_ID: problem.x.id}

        # Format constraints
        #
        # SCS requires constraints to be specified in the following order:
        # 1. zero cone
        # 2. non-negative orthant
        # 3. soc
        # 4. psd
        # 5. exponential
        # 6. three-dimensional power cones
        if not problem.formatted:
            problem = self.format_constraints(problem, self.EXP_CONE_ORDER)
        data[s.PARAM_PROB] = problem
        data[self.DIMS] = problem.cone_dims
        inv_data[self.DIMS] = problem.cone_dims

        constr_map = problem.constr_map
        inv_data[self.EQ_CONSTR] = constr_map[Zero]
        inv_data[self.NEQ_CONSTR] = constr_map[NonNeg] + constr_map[SOC] + \
            constr_map[PSD] + constr_map[ExpCone] + constr_map[PowCone3D]
        return problem, data, inv_data

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        problem, data, inv_data = self._prepare_data_and_inv_data(problem)

        # Apply parameter values.
        # Obtain A, b such that Ax + s = b, s \in cones.
        c, d, A, b = problem.apply_parameters()
        data[s.C] = c
        inv_data[s.OFFSET] = d
        data[s.A] = -A
        data[s.B] = b
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
        attr[s.EXTRA_STATS] = solution

        if status in s.SOLUTION_PRESENT:
            primal_val = solution["info"]["pobj"]
            opt_val = primal_val + inverse_data[s.OFFSET]
            # TODO expand primal and dual variables from lower triangular to full.
            # TODO but this makes map from solution to variables not a slice.
            primal_vars = {
                inverse_data[SCS.VAR_ID]: solution["x"]
            }
            eq_dual_vars = utilities.get_dual_values(
                solution["y"][:inverse_data[ConicSolver.DIMS].zero],
                self.extract_dual_value,
                inverse_data[SCS.EQ_CONSTR]
            )
            ineq_dual_vars = utilities.get_dual_values(
                solution["y"][inverse_data[ConicSolver.DIMS].zero:],
                self.extract_dual_value,
                inverse_data[SCS.NEQ_CONSTR]
            )
            dual_vars = {}
            dual_vars.update(eq_dual_vars)
            dual_vars.update(ineq_dual_vars)
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

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
        solver_opts["eps"] = solver_opts.get("eps", 1e-4)

        results = scs.solve(args, cones, verbose=verbose, **solver_opts)
        status = self.STATUS_MAP[results["info"]["status"]]

        if (status == s.OPTIMAL_INACCURATE and
                "acceleration_lookback" not in solver_opts):
            # anderson acceleration is sometimes unstable; retry without it
            results = scs.solve(
                args,
                cones,
                verbose=verbose,
                acceleration_lookback=0,
                **solver_opts)

        if solver_cache is not None:
            solver_cache[self.name()] = results
        return results
