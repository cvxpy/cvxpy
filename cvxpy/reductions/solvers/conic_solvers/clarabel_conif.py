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
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import SOC, ExpCone, PowCone3D, PowConeND, SvecPSD
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.solver_inverse_data import SolverInverseData
from cvxpy.utilities.citations import CITATION_DICT
from cvxpy.utilities.psd_utils import TriangleKind


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

    for pow in cone_dims.pnd:
        # TODO: On the right hand side, we may want to
        # extend to support higher dim values for z
        # instead of hardcoding 1.
        cones.append(clarabel.GenPowerConeT(pow, 1))
    return cones



class CLARABEL(ConicSolver):
    """An interface for the Clarabel solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS \
        + [SOC, ExpCone, PowCone3D, SvecPSD, PowConeND]
    PSD_TRIANGLE_KIND = TriangleKind.UPPER
    PSD_SQRT2_SCALING = True

    # Status messages from clarabel.
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
    UNSOLVED = "Unsolved"
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
                    UNSOLVED: s.SOLVER_ERROR,
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

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        attr = {}
        status_map = self.STATUS_MAP.copy()

        # if accept unknown was specified and solution is present, then an insufficient progress
        # status will be mapped to OPTIMAL_INACCURATE.
        if isinstance(inverse_data, SolverInverseData) and\
            CLARABEL.ACCEPT_UNKNOWN in inverse_data.solver_options and\
            solution.x is not None and solution.z is not None:
            status_map["InsufficientProgress"] = s.OPTIMAL_INACCURATE
        status = status_map.get(str(solution.status), s.SOLVER_ERROR)
        attr[s.SOLVE_TIME] = solution.solve_time
        attr[s.NUM_ITERS] = solution.iterations
        # more detailed statistics here when available
        # attr[s.EXTRA_STATS] = solution.extra.FOO

        if solution.z is not None:
            zero_idx = inverse_data[ConicSolver.DIMS].zero
            eq_dual_vars = utilities.get_dual_values(
                solution.z[:zero_idx],
                utilities.extract_dual_value,
                inverse_data[self.EQ_CONSTR]
            )
            ineq_dual_vars = utilities.get_dual_values(
                solution.z[zero_idx:],
                utilities.extract_dual_value,
                inverse_data[self.NEQ_CONSTR]
            )
            dual_vars = eq_dual_vars | ineq_dual_vars
        else:
            dual_vars = {}

        if status in s.SOLUTION_PRESENT:
            primal_val = solution.obj_val
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {
                inverse_data[self.VAR_ID]: solution.x
            }
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr, dual_vars)

    @staticmethod
    def parse_solver_opts(verbose, opts, settings=None):
        import clarabel

        if settings is None:
            settings = clarabel.DefaultSettings()

        settings.verbose = verbose

        keys = list(opts.keys())

        # use_quad_obj is only for canonicalization.
        if "use_quad_obj" in keys:
            keys.remove("use_quad_obj")
        if CLARABEL.ACCEPT_UNKNOWN in keys:
            keys.remove(CLARABEL.ACCEPT_UNKNOWN)

        for opt in keys:
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

            _settings = self.parse_solver_opts(verbose, solver_opts)
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
                newsettings = self.parse_solver_opts(verbose, solver_opts, oldsettings)

                # this overwrites all data in the solver but will not
                # reallocate internal memory.  Could be faster if it
                # were known which terms (or partial terms) have changed.
                try:
                    _solver.update(P=P, q=q, A=A, b=b, settings=newsettings)
                    return _solver
                except Exception:
                    # If sparsity pattern or dimensions changed, update() fails.
                    # Return None to trigger a full new_solver() re-initialization.
                    return None

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
