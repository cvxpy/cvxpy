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
from cvxpy.constraints import SOC, ComplexPSD, ExpCone, PowCone3D, SvecPSD
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    ConicSolver,
)
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    dims_to_solver_dict as dims_to_solver_dict_default,
)
from cvxpy.utilities.citations import CITATION_DICT
from cvxpy.utilities.psd_utils import TriangleKind
from cvxpy.utilities.versioning import Version
from cvxpy.utilities.warn import warn


def dims_to_solver_dict(cone_dims):
    cones = dims_to_solver_dict_default(cone_dims)

    import scs
    if Version(scs.__version__) >= Version('3.0.0'):
        cones['z'] = cones.pop('f')  # renamed to 'z' in SCS 3.0.0
    # SCS uses 'cs' for Hermitian PSD cone dimensions (same as CVXPY)
    return cones


def _cvec_to_complex_full(cvec, n):
    """Expand n^2 cvec to flat complex array of length n^2.

    cvec format: for each column j, diagonal entry S_jj is stored as-is,
    then for each i > j, two entries [sqrt(2)*Re(S_ij), sqrt(2)*Im(S_ij)].

    Returns a column-major flat complex array of the full n x n matrix.
    """
    real = np.zeros((n, n))
    imag = np.zeros((n, n))
    idx = 0
    for j in range(n):
        real[j, j] = cvec[idx]
        idx += 1
        for i in range(j + 1, n):
            real[i, j] = real[j, i] = cvec[idx] / np.sqrt(2)
            idx += 1
            imag[i, j] = cvec[idx] / np.sqrt(2)
            imag[j, i] = -imag[i, j]
            idx += 1
    return np.reshape(real + 1j * imag, n * n, order='F')
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
        + [SOC, ExpCone, SvecPSD, PowCone3D, ComplexPSD]
    REQUIRES_CONSTR = True
    PSD_TRIANGLE_KIND = TriangleKind.LOWER
    PSD_SQRT2_SCALING = True

    # Map of SCS status value to CVXPY status.
    STATUS_MAP = {1: s.OPTIMAL,
                  2: s.OPTIMAL_INACCURATE,
                  -1: s.UNBOUNDED,
                  -6: s.UNBOUNDED_INACCURATE,
                  -2: s.INFEASIBLE,
                  -7: s.INFEASIBLE_INACCURATE,
                  -4: s.SOLVER_ERROR,           # Failed
                  -3: s.SOLVER_ERROR,           # Indeterminate
                  -5: s.SOLVER_ERROR}           # SIGINT

    # Order of exponential cone arguments for solver.
    EXP_CONE_ORDER = [0, 1, 2]

    ACCELERATION_RETRY_MESSAGE = """
    CVXPY has just called the numerical solver SCS (version %s),
    which could not accurately solve the problem with the provided solver
    options. No value was specified for the SCS option called
    "acceleration_lookback". That option often has a major impact on
    whether this version of SCS converges to an accurate solution.

    We will try to solve the problem again by setting acceleration_lookback = 0.
    To avoid this error in the future we recommend installing SCS version 3.0
    or higher.

    More information on SCS options can be found at the following URL:
    https://www.cvxgrp.org/scs/api/settings.html
    """

    def name(self):
        """The name of the solver.
        """
        return s.SCS

    def import_solver(self) -> None:
        """Imports the solver.
        """
        import scs  # noqa F401

    def supports_quad_obj(self) -> bool:
        """SCS >= 3.0.0 supports a quadratic objective.
        """
        import scs
        return Version(scs.__version__) >= Version('3.0.0')

    @staticmethod
    def complex_psd_format_mat(constr):
        """Return a linear operator to multiply by ComplexPSD constraint coefficients.

        Maps stacked [R_flat, I_flat] (2*n^2 entries) to SCS cvec format (n^2 entries).

        SCS cvec format for k x k Hermitian matrix S:
          cvec(S) = (S_11, sqrt(2)*Re(S_21), sqrt(2)*Im(S_21), ...,
                     sqrt(2)*Re(S_k1), sqrt(2)*Im(S_k1), S_22, ...)

        The input is [vec(R), vec(I)] where R is the real part and I is the
        imaginary part (both column-major). We apply Hermitianization:
          - Diagonal (j,j): output = R[j,j]
          - Off-diagonal (i,j) i>j: Re entry = sqrt(2)/2 * (R[i,j] + R[j,i])
                                     Im entry = sqrt(2)/2 * (I[i,j] - I[j,i])
        """
        n = constr.args[0].shape[0]
        n2 = n * n
        cvec_size = n * n  # output size

        rows = []
        cols = []
        vals = []
        out_idx = 0

        for j in range(n):
            # Diagonal entry: output = R[j,j]
            r_idx = j * n + j  # column-major index in R
            rows.append(out_idx)
            cols.append(r_idx)
            vals.append(1.0)
            out_idx += 1

            for i in range(j + 1, n):
                # Real part of off-diagonal: sqrt(2)/2 * (R[i,j] + R[j,i])
                r_ij = j * n + i  # R[i,j] in column-major
                r_ji = i * n + j  # R[j,i] in column-major
                s2_2 = np.sqrt(2) / 2.0

                rows.append(out_idx)
                cols.append(r_ij)
                vals.append(s2_2)

                rows.append(out_idx)
                cols.append(r_ji)
                vals.append(s2_2)
                out_idx += 1

                # Imaginary part of off-diagonal: sqrt(2)/2 * (I[i,j] - I[j,i])
                i_ij = n2 + j * n + i  # I[i,j] in column-major, offset by n^2
                i_ji = n2 + i * n + j  # I[j,i] in column-major, offset by n^2

                rows.append(out_idx)
                cols.append(i_ij)
                vals.append(s2_2)

                rows.append(out_idx)
                cols.append(i_ji)
                vals.append(-s2_2)
                out_idx += 1

        shape = (cvec_size, 2 * n2)
        return sp.csc_array((vals, (rows, cols)), shape)

    @staticmethod
    def extract_dual_value(result_vec, offset, constraint):
        """Extracts the dual value for constraint starting at offset.

        Special cases ComplexPSD constraints, as per the SCS specification.
        """
        if isinstance(constraint, ComplexPSD):
            dim = constraint.args[0].shape[0]
            cvec_dim = dim * dim
            new_offset = offset + cvec_dim
            cvec = result_vec[offset:new_offset]
            full = _cvec_to_complex_full(cvec, dim)
            return full, new_offset
        else:
            return utilities.extract_dual_value(result_vec, offset,
                                                constraint)

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        return super(SCS, self).apply(problem)

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        import scs
        attr = {}
        # SCS versions 1.*, SCS 2.*
        if Version(scs.__version__) < Version('3.0.0'):
            status = self.STATUS_MAP[solution["info"]["statusVal"]]
            attr[s.SOLVE_TIME] = solution["info"]["solveTime"] / 1000
            attr[s.SETUP_TIME] = solution["info"]["setupTime"] / 1000

        # SCS version 3.*
        else:
            status = self.STATUS_MAP[solution["info"]["status_val"]]
            attr[s.SOLVE_TIME] = solution["info"]["solve_time"] / 1000
            attr[s.SETUP_TIME] = solution["info"]["setup_time"] / 1000

        attr[s.NUM_ITERS] = solution["info"]["iter"]
        attr[s.EXTRA_STATS] = solution

        zero_idx = inverse_data[ConicSolver.DIMS].zero
        eq_dual_vars = utilities.get_dual_values(
            solution["y"][:zero_idx],
            SCS.extract_dual_value,
            inverse_data[SCS.EQ_CONSTR]
        )
        ineq_dual_vars = utilities.get_dual_values(
            solution["y"][zero_idx:],
            SCS.extract_dual_value,
            inverse_data[SCS.NEQ_CONSTR]
        )
        dual_vars = eq_dual_vars | ineq_dual_vars

        if status in s.SOLUTION_PRESENT:
            primal_val = solution["info"]["pobj"]
            opt_val = primal_val + inverse_data[s.OFFSET]
            # TODO expand primal and dual variables from lower triangular to full.
            # TODO but this makes map from solution to variables not a slice.
            primal_vars = {
                inverse_data[SCS.VAR_ID]: solution["x"]
            }
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr, dual_vars)

    @staticmethod
    def parse_solver_options(solver_opts):
        import scs
        if Version(scs.__version__) < Version('3.0.0'):
            if "eps_abs" in solver_opts or "eps_rel" in solver_opts:
                # Take the min of eps_rel and eps_abs to be eps
                solver_opts["eps"] = min(solver_opts.get("eps_abs", 1),
                                         solver_opts.get("eps_rel", 1))
            else:
                # Default to eps = 1e-4 instead of 1e-3.
                solver_opts["eps"] = solver_opts.get("eps", 1e-4)
        else:
            if "eps" in solver_opts:  # eps replaced by eps_abs, eps_rel
                solver_opts["eps_abs"] = solver_opts["eps"]
                solver_opts["eps_rel"] = solver_opts["eps"]
                del solver_opts["eps"]
            else:
                solver_opts['eps_abs'] = solver_opts.get('eps_abs', 1e-5)
                solver_opts['eps_rel'] = solver_opts.get('eps_rel', 1e-5)
        # use_quad_obj is only for canonicalization.
        if "use_quad_obj" in solver_opts:
            del solver_opts["use_quad_obj"]
        return solver_opts

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
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
        scs_version = Version(scs.__version__)
        args = {"A": data[s.A], "b": data[s.B], "c": data[s.C]}
        if s.P in data:
            args["P"] = data[s.P]
        if warm_start and solver_cache is not None and \
                self.name() in solver_cache:
            args["x"] = solver_cache[self.name()]["x"]
            args["y"] = solver_cache[self.name()]["y"]
            args["s"] = solver_cache[self.name()]["s"]
        cones = dims_to_solver_dict(data[ConicSolver.DIMS])

        def solve(_solver_opts):
            if scs_version.major < 3:
                _results = scs.solve(args, cones, verbose=verbose, **_solver_opts)
                _status = self.STATUS_MAP[_results["info"]["statusVal"]]
            else:
                _results = scs.solve(args, cones, verbose=verbose, **_solver_opts)
                _status = self.STATUS_MAP[_results["info"]["status_val"]]
            return _results, _status

        solver_opts = SCS.parse_solver_options(solver_opts)
        results, status = solve(solver_opts)
        if (status in s.INACCURATE and scs_version.major == 2
                and "acceleration_lookback" not in solver_opts):
            warn(SCS.ACCELERATION_RETRY_MESSAGE % str(scs_version))
            retry_opts = solver_opts.copy()
            retry_opts["acceleration_lookback"] = 0
            results, status = solve(retry_opts)

        if solver_cache is not None and status == s.OPTIMAL:
            solver_cache[self.name()] = results
        return results

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["SCS"]
