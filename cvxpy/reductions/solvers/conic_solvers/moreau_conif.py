"""
Copyright 2025, the CVXPY Authors

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
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.solver_inverse_data import SolverInverseData
from cvxpy.utilities.citations import CITATION_DICT


def dims_to_solver_cones(cone_dims):
    """Convert CVXPY cone dimensions to Moreau cone specification.

    Parameters
    ----------
    cone_dims : ConeDims
        CVXPY cone dimensions object

    Returns
    -------
    moreau.Cones
        Moreau cone specification
    """
    import moreau

    # Moreau does not support generalized power cones yet
    if cone_dims.pnd:
        raise ValueError("Moreau does not support generalized power cones (PowConeND)")

    cones = moreau.Cones(
        num_zero_cones=cone_dims.zero,
        num_nonneg_cones=cone_dims.nonneg,
        so_cone_dims=list(cone_dims.soc),
        num_exp_cones=cone_dims.exp,
        power_alphas=list(cone_dims.p3d),
        psd_dims=list(cone_dims.psd) if cone_dims.psd else [],
    )

    return cones


class MOREAU(ConicSolver):
    """An interface for the Moreau GPU solver.

    Moreau is a GPU-accelerated conic optimization solver based on the
    Clarabel interior-point algorithm, designed for high-throughput batch solving.
    """

    # Solver capabilities
    MIP_CAPABLE = False
    BOUNDED_VARIABLES = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, ExpCone, PowCone3D, PSD]

    # Status messages from Moreau (based on solver/status.hpp)
    SOLVED = "Solved"
    PRIMAL_INFEASIBLE = "PrimalInfeasible"
    DUAL_INFEASIBLE = "DualInfeasible"
    ALMOST_SOLVED = "AlmostSolved"
    ALMOST_PRIMAL_INFEASIBLE = "AlmostPrimalInfeasible"
    ALMOST_DUAL_INFEASIBLE = "AlmostDualInfeasible"
    MAX_ITERATIONS = "MaxIterations"
    MAX_TIME = "MaxTime"
    NUMERICAL_ERROR = "NumericalError"
    INSUFFICIENT_PROGRESS = "InsufficientProgress"
    ACCEPT_UNKNOWN = "accept_unknown"

    STATUS_MAP = {
        SOLVED: s.OPTIMAL,
        PRIMAL_INFEASIBLE: s.INFEASIBLE,
        DUAL_INFEASIBLE: s.UNBOUNDED,
        ALMOST_SOLVED: s.OPTIMAL_INACCURATE,
        ALMOST_PRIMAL_INFEASIBLE: s.INFEASIBLE_INACCURATE,
        ALMOST_DUAL_INFEASIBLE: s.UNBOUNDED_INACCURATE,
        MAX_ITERATIONS: s.USER_LIMIT,
        MAX_TIME: s.USER_LIMIT,
        NUMERICAL_ERROR: s.SOLVER_ERROR,
        INSUFFICIENT_PROGRESS: s.SOLVER_ERROR,
    }

    # Order of exponential cone arguments for solver
    EXP_CONE_ORDER = [0, 1, 2]

    def name(self):
        """The name of the solver."""
        return "MOREAU"

    def import_solver(self) -> None:
        """Imports the solver."""
        import moreau  # noqa F401

    def supports_quad_obj(self) -> bool:
        """Moreau supports quadratic objective with conic constraints."""
        return True

    @staticmethod
    def psd_format_mat(constr):
        """Convert PSD constraint to upper-triangular svec format.

        Moreau (Clarabel-based) uses upper-triangular column-major svec:
        for col in 0..n, for row in 0..=col: (row, col)
        giving ordering (0,0), (0,1), (1,1), (0,2), (1,2), (2,2), ...

        Off-diagonal coefficients are scaled by sqrt(2), and the operator
        is applied to the symmetric part of the constrained expression.
        """
        rows = cols = constr.expr.shape[0]
        entries = rows * (cols + 1) // 2

        row_arr = np.arange(0, entries)
        upper_diag_indices = np.triu_indices(rows)
        col_arr = np.sort(np.ravel_multi_index(upper_diag_indices,
                                               (rows, cols), order='F'))

        val_arr = np.zeros((rows, cols))
        val_arr[upper_diag_indices] = np.sqrt(2)
        np.fill_diagonal(val_arr, 1.0)
        val_arr = np.ravel(val_arr, order='F')
        val_arr = val_arr[np.nonzero(val_arr)]

        shape = (entries, rows * cols)
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

        Special cases PSD constraints: dual is stored in upper-triangular
        svec format (Moreau/Clarabel convention) and must be expanded to
        the full symmetric matrix.
        """
        if isinstance(constraint, PSD):
            dim = constraint.shape[0]
            tri_dim = dim * (dim + 1) // 2
            new_offset = offset + tri_dim
            svec = result_vec[offset:new_offset]
            # Unpack upper-triangular svec: col-major, row <= col
            full = np.zeros((dim, dim))
            idx = 0
            for col in range(dim):
                for row in range(col + 1):
                    if row == col:
                        full[row, col] = svec[idx]
                    else:
                        full[row, col] = svec[idx] / np.sqrt(2)
                        full[col, row] = full[row, col]
                    idx += 1
            return full, new_offset
        else:
            return utilities.extract_dual_value(result_vec, offset, constraint)

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data."""
        attr = {}
        status_map = self.STATUS_MAP.copy()

        # Handle accept_unknown option
        if (
            isinstance(inverse_data, SolverInverseData)
            and MOREAU.ACCEPT_UNKNOWN in inverse_data.solver_options
            and solution.x is not None
            and solution.z is not None
        ):
            status_map["InsufficientProgress"] = s.OPTIMAL_INACCURATE

        status = status_map.get(str(solution.status), s.SOLVER_ERROR)
        attr[s.SOLVE_TIME] = solution.solve_time
        attr[s.SETUP_TIME] = solution.setup_time
        attr[s.NUM_ITERS] = solution.iterations

        if status in s.SOLUTION_PRESENT:
            primal_val = solution.obj_val
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: solution.x}
            eq_dual_vars = utilities.get_dual_values(
                solution.z[: inverse_data[ConicSolver.DIMS].zero],
                self.extract_dual_value,
                inverse_data[self.EQ_CONSTR],
            )
            ineq_dual_vars = utilities.get_dual_values(
                solution.z[inverse_data[ConicSolver.DIMS].zero :],
                self.extract_dual_value,
                inverse_data[self.NEQ_CONSTR],
            )
            dual_vars = {}
            dual_vars.update(eq_dual_vars)
            dual_vars.update(ineq_dual_vars)
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    @staticmethod
    def handle_options(verbose: bool, solver_opts: dict):
        """Handle user-specified solver options.

        Parameters
        ----------
        verbose : bool
            Enable verbose output
        solver_opts : dict
            Solver-specific options

        Returns
        -------
        tuple
            (settings, processed_opts) where settings is moreau.Settings and
            processed_opts contains accept_unknown
        """
        import moreau

        solver_opts = solver_opts.copy() if solver_opts else {}

        # Extract cvxpy-specific options
        processed_opts = {}
        processed_opts["accept_unknown"] = solver_opts.pop("accept_unknown", False)

        # Extract device (now part of Settings)
        device = solver_opts.pop("device", "auto")

        # Remove use_quad_obj (handled by reduction chain, not solver)
        solver_opts.pop("use_quad_obj", None)

        # Create Settings with device and verbose
        settings = moreau.Settings(
            device=device,
            verbose=verbose,
        )

        # Handle ipm_settings: accept dict or moreau.IPMSettings
        ipm_settings = solver_opts.pop("ipm_settings", None)
        if ipm_settings is not None:
            if isinstance(ipm_settings, dict):
                ipm_settings = moreau.IPMSettings(**ipm_settings)
            settings.ipm_settings = ipm_settings

        # Apply all remaining options directly to moreau.Settings
        for opt, value in solver_opts.items():
            try:
                setattr(settings, opt, value)
            except AttributeError as e:
                raise TypeError(f"Moreau: unrecognized solver setting '{opt}'.") from e
            except TypeError as e:
                raise TypeError(f"Moreau: incorrect type for setting '{opt}'.") from e

        return settings, processed_opts

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        warm_start : bool
            Whether to warm_start Moreau (not currently supported).
        verbose : bool
            Control the verbosity.
        solver_opts : dict
            Moreau-specific solver options.
        solver_cache : dict, optional
            Cache for solver objects (not currently used).

        Returns
        -------
        Solution object from Moreau solver.
        """
        import moreau

        A = data[s.A]
        b = data[s.B]
        q = data[s.C]

        if s.P in data:
            P = data[s.P]
        else:
            nvars = q.size
            # Create empty sparse matrix with proper structure
            P = sp.csr_array((nvars, nvars))

        # Convert to CSR format
        P = P.tocsr()
        A = A.tocsr()

        # Convert cone dimensions
        cones = dims_to_solver_cones(data[ConicSolver.DIMS])

        # Handle options (device is now part of Settings)
        settings, processed_opts = self.handle_options(verbose, solver_opts or {})

        # Create solver with all problem data in constructor
        solver = moreau.Solver(
            P=P,
            q=q.astype(np.float64),
            A=A,
            b=b.astype(np.float64),
            cones=cones,
            settings=settings,
        )

        # Solve (no arguments - all data was provided in constructor)
        solution = solver.solve()
        info = solver.info  # Metadata is on solver.info after solve()

        return MoreauSolution(solution, info)

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.

        Returns
        -------
        str
            BibTeX citation
        """
        return CITATION_DICT["MOREAU"]


class MoreauSolution:
    """Wrapper combining solution vectors with solver metadata.

    Moreau separates solution vectors (x, s, z from solver.solve()) from
    metadata (status, timing, iterations from solver.info). This wrapper
    combines them into a single object for use in invert().
    """

    def __init__(self, sol, info):
        self.x = sol.x
        self.s = sol.s
        self.z = sol.z
        self.status = info.status.name
        self.iterations = info.iterations
        self.solve_time = info.solve_time
        self.setup_time = info.setup_time
        self.obj_val = info.obj_val
