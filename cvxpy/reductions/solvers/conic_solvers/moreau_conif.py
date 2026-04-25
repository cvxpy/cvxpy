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
from cvxpy.constraints import SOC, ExpCone, PowCone3D
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.solver_inverse_data import SolverInverseData
from cvxpy.utilities.citations import CITATION_DICT


def _moreau_supports_x_cones() -> bool:
    """True if the installed Moreau build exposes the x_cones API.

    Older Moreau lacks ``XConeSpec``; on those installs ``x_cone_kinds()``
    returns an empty set and the chain skips ExtractIdentityCones.
    """
    try:
        import moreau
    except ImportError:
        return False
    return hasattr(moreau, 'XConeSpec')


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

    # Moreau does not support PSD cones yet
    if cone_dims.psd:
        raise ValueError("Moreau does not support PSD cones")

    # Moreau does not support generalized power cones yet
    if cone_dims.pnd:
        raise ValueError("Moreau does not support generalized power cones (PowConeND)")

    cones = moreau.Cones(
        num_zero_cones=cone_dims.zero,
        num_nonneg_cones=cone_dims.nonneg,
        so_cone_dims=list(cone_dims.soc),
        num_exp_cones=cone_dims.exp,
        power_alphas=list(cone_dims.p3d),
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
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, ExpCone, PowCone3D]

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

    def x_cone_kinds(self) -> frozenset:
        """Direct-x cone kinds Moreau accepts via XConeSpec.

        Empty when the installed Moreau is too old to know about
        x_cones, so ExtractIdentityCones becomes a no-op.
        """
        if not _moreau_supports_x_cones():
            return frozenset()
        return frozenset({'nonneg', 'soc'})

    def apply(self, problem):
        """Forward ``problem.x_cones`` (set by ExtractIdentityCones) into
        the solver's data and inverse-data dicts; otherwise behaves
        exactly as ConicSolver.apply.
        """
        data, inv_data = super().apply(problem)
        if problem.x_cones:
            data['x_cones'] = problem.x_cones
            inv_data['x_cones'] = problem.x_cones
        return data, inv_data

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
                utilities.extract_dual_value,
                inverse_data[self.EQ_CONSTR],
            )
            ineq_dual_vars = utilities.get_dual_values(
                solution.z[inverse_data[ConicSolver.DIMS].zero :],
                utilities.extract_dual_value,
                inverse_data[self.NEQ_CONSTR],
            )
            dual_vars = {}
            dual_vars.update(eq_dual_vars)
            dual_vars.update(ineq_dual_vars)
            # x_cone duals (computed in solve_via_data from the KKT
            # residual q + A.T z) are keyed by their original
            # constraint id and slot in directly.
            xcone_duals = getattr(solution, 'x_cone_duals', None) or {}
            dual_vars.update(xcone_duals)
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

        # Direct-x cones from ExtractIdentityCones: subvectors of the
        # primal variable that ride the XConeSpec path instead of
        # being slack-side cones.
        x_cones_meta = data.get('x_cones', []) or []
        if x_cones_meta:
            cones.x_cones = [
                moreau.XConeSpec(kind=kind, indices=list(indices))
                for kind, indices, _constr_id in x_cones_meta
            ]

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

        wrapped = MoreauSolution(solution, info)
        # Recover x_cone duals from the KKT residual:
        #   μ_block = (q + A.T z)[x_indices]
        # which lives in K* (matches CVXPY's convention for NonNeg/SOC
        # duals).  See cvxpy/reductions/cone2cone/extract_identity_cones.py.
        if x_cones_meta and solution.z is not None:
            kkt_resid = q + A.T @ solution.z
            wrapped.x_cone_duals = {
                constr_id: kkt_resid[list(indices)]
                for _kind, indices, constr_id in x_cones_meta
            }
        else:
            wrapped.x_cone_duals = {}
        return wrapped

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
