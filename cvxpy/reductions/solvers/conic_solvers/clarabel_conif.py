"""
Copyright 2022, the CVXPY Authors

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
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.citations import CITATION_DICT


def dims_to_solver_cones(cone_dims):

    import clarabel
    cones = []

    # assume that constraints are presented
    # in the preferred ordering of SCS.

    if cone_dims.zero > 0:
        cones.append(clarabel.ZeroConeT(cone_dims.zero))

    if cone_dims.nonneg > 0:
        cones.append(clarabel.NonnegativeConeT(cone_dims.nonneg))

    for dim in cone_dims.soc:
        cones.append(clarabel.SecondOrderConeT(dim))

    for dim in cone_dims.psd:
        cones.append(clarabel.PSDTriangleConeT(dim))

    for _ in range(cone_dims.exp):
        cones.append(clarabel.ExponentialConeT())

    for pow in cone_dims.p3d:
        cones.append(clarabel.PowerConeT(pow))
    return cones


def triu_to_full(upper_tri, n):
    """Expands n*(n+1)//2 upper triangular to full matrix, scaling
    off diagonals by 1/sqrt(2).   This is similar to the SCS behaviour,
    but the upper triangle is used.

    Parameters
    ----------
    upper_tri : numpy.ndarray
        A NumPy array representing the upper triangular part of the
        matrix, stacked in column-major order.
    n : int
        The number of rows (columns) in the full square matrix.

    Returns
    -------
    numpy.ndarray
        A 2-dimensional ndarray that is the scaled expansion of the upper
        triangular array.

    Notes
    -----
    As in the related SCS function, the function below appears to have
    triu/tril confused but is nevertheless correct.

    """
    full = np.zeros((n, n))
    full[np.tril_indices(n)] = upper_tri
    full += full.T
    full[np.diag_indices(n)] /= 2
    full[np.tril_indices(n, k=-1)] /= np.sqrt(2)
    full[np.triu_indices(n, k=1)] /= np.sqrt(2)
    return np.reshape(full, n*n, order="F")


def clarabel_psdvec_to_psdmat(vec: Expression, indices: np.ndarray) -> Expression:
    """
    Return "V" so that "vec[indices] belongs to the Clarabel PSDTriangleCone"
    can be written in natural cvxpy syntax as "V >> 0".

    Parameters
    ----------
    vec : cvxpy.expressions.expression.Expression
        Must have ``vec.is_affine() == True``.
    indices : ndarray
        Contains nonnegative integers, which can index into ``vec``.

    Notes
    -----
    This function is similar to ``triu_to_full``, which is also found
    in this file. The difference is that this function works without
    indexed assignment ``mat[i,j] = expr``. Such indexed assignment
    cannot be used, because this function builds a cvxpy Expression,
    rather than a numpy ndarray.
    """
    n = int(np.sqrt(indices.size * 2))
    rows, cols = np.tril_indices(n)   # tril here not an error
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


class CLARABEL(ConicSolver):
    """An interface for the Clarabel solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS \
        + [SOC, ExpCone, PowCone3D, PSD]

    STATUS_MAP = {
                    "Solved": s.OPTIMAL,
                    "PrimalInfeasible": s.INFEASIBLE,
                    "DualInfeasible": s.UNBOUNDED,
                    "AlmostSolved": s.OPTIMAL_INACCURATE,
                    "AlmostPrimalInfeasible": s.INFEASIBLE_INACCURATE,
                    "AlmostDualInfeasible": s.UNBOUNDED_INACCURATE,
                    "MaxIterations": s.USER_LIMIT,
                    "MaxTime": s.USER_LIMIT,
                    "NumericalError": s.SOLVER_ERROR,
                    "InsufficientProgress": s.SOLVER_ERROR
                }

    # Order of exponential cone arguments for solver.
    EXP_CONE_ORDER = [0, 1, 2]

    def name(self):
        """The name of the solver.
        """
        return 'CLARABEL'

    def import_solver(self) -> None:
        """Imports the solver.
        """
        import clarabel  # noqa F401

    def supports_quad_obj(self) -> bool:
        """Clarabel supports quadratic objective with any combination
        of conic constraints.
        """
        return True

    @staticmethod
    def psd_format_mat(constr):
        """Return a linear operator to multiply by PSD constraint coefficients.

        Special cases PSD constraints, as Clarabel expects constraints to be
        imposed on the upper triangular part of the variable matrix with
        symmetric scaling (i.e. off-diagonal sqrt(2) scalinig) applied.

        """
        rows = cols = constr.expr.shape[0]
        entries = rows * (cols + 1)//2

        row_arr = np.arange(0, entries)

        upper_diag_indices = np.triu_indices(rows)
        col_arr = np.sort(np.ravel_multi_index(upper_diag_indices,
                                               (rows, cols),
                                               order='F'))

        val_arr = np.zeros((rows, cols))
        val_arr[upper_diag_indices] = np.sqrt(2)
        np.fill_diagonal(val_arr, 1.0)
        val_arr = np.ravel(val_arr, order='F')
        val_arr = val_arr[np.nonzero(val_arr)]

        shape = (entries, rows*cols)
        scaled_upper_tri = sp.csc_array((val_arr, (row_arr, col_arr)), shape)

        idx = np.arange(rows * cols)
        val_symm = 0.5 * np.ones(2 * rows * cols)
        K = idx.reshape((rows, cols))
        row_symm = np.append(idx, np.ravel(K, order='F'))
        col_symm = np.append(idx, np.ravel(K.T, order='F'))
        symm_matrix = sp.csc_array((val_symm, (row_symm, col_symm)))

        return scaled_upper_tri @ symm_matrix

    @staticmethod
    def extract_dual_value(result_vec, offset, constraint):
        """Extracts the dual value for constraint starting at offset.
        """

        # special case: PSD constraints treated internally in
        # svec (scaled triangular) form
        if isinstance(constraint, PSD):
            dim = constraint.shape[0]
            upper_tri_dim = dim * (dim + 1) >> 1
            new_offset = offset + upper_tri_dim
            upper_tri = result_vec[offset:new_offset]
            full = triu_to_full(upper_tri, dim)
            return full, new_offset

        else:
            return utilities.extract_dual_value(result_vec, offset, constraint)

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """

        attr = {}
        status = self.STATUS_MAP[str(solution.status)]
        attr[s.SOLVE_TIME] = solution.solve_time
        attr[s.NUM_ITERS] = solution.iterations
        # more detailed statistics here when available
        # attr[s.EXTRA_STATS] = solution.extra.FOO

        if status in s.SOLUTION_PRESENT:
            primal_val = solution.obj_val
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {
                inverse_data[CLARABEL.VAR_ID]: solution.x
            }
            eq_dual_vars = utilities.get_dual_values(
                solution.z[:inverse_data[ConicSolver.DIMS].zero],
                self.extract_dual_value,
                inverse_data[CLARABEL.EQ_CONSTR]
            )
            ineq_dual_vars = utilities.get_dual_values(
                solution.z[inverse_data[ConicSolver.DIMS].zero:],
                self.extract_dual_value,
                inverse_data[CLARABEL.NEQ_CONSTR]
            )
            dual_vars = {}
            dual_vars.update(eq_dual_vars)
            dual_vars.update(ineq_dual_vars)
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    @staticmethod
    def parse_solver_opts(verbose, opts, settings=None):
        import clarabel

        if settings is None:
            settings = clarabel.DefaultSettings()

        settings.verbose = verbose

        # use_quad_obj is only for canonicalization.
        if "use_quad_obj" in opts:
            del opts["use_quad_obj"]

        for opt in opts.keys():
            try:
                settings.__setattr__(opt, opts[opt])
            except TypeError as e:
                raise TypeError(f"Clarabel: Incorrect type for setting '{opt}'.") from e
            except AttributeError as e:
                raise TypeError(f"Clarabel: unrecognized solver setting '{opt}'.") from e

        return settings

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        warm_start : Bool
            Whether to warm_start Clarabel.
        verbose : Bool
            Control the verbosity.
        solver_opts : dict
            Clarabel-specific solver options.

        Returns
        -------
        The result returned by a call to clarabel.solve().
        """
        import clarabel

        A = data[s.A]
        b = data[s.B]
        q = data[s.C]

        if s.P in data:
            P = data[s.P]
        else:
            nvars = q.size
            P = sp.csc_array((nvars, nvars))

        P = sp.triu(P).tocsc()

        cones = dims_to_solver_cones(data[ConicSolver.DIMS])

        def new_solver():

            _settings = CLARABEL.parse_solver_opts(verbose, solver_opts)
            _solver = clarabel.DefaultSolver(P, q, A, b, cones, _settings)
            return _solver

        def updated_solver():

            if (not warm_start) or (solver_cache is None) or (self.name() not in solver_cache):
                return None

            _solver = solver_cache[self.name()]

            if not hasattr(_solver, "update"):
                return None
            elif not _solver.is_data_update_allowed():
                # disallow when presolve or chordal decomposition is used
                return None
            else:
                # current internal settings, to be updated if needed
                oldsettings = _solver.get_settings()
                newsettings = CLARABEL.parse_solver_opts(verbose, solver_opts, oldsettings)

                # this overwrites all data in the solver but will not
                # reallocate internal memory.  Could be faster if it
                # were known which terms (or partial terms) have changed.
                # Will error ail if dimensions are sparsity has changed
                _solver.update(P=P, q=q, A=A, b=b, settings=newsettings)
                return _solver

        # Try to get cached data
        solver = updated_solver()

        if solver is None:
            solver = new_solver()

        results = solver.solve()

        if solver_cache is not None:
            solver_cache[self.name()] = solver

        return results
    
    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["CLARABEL"]
