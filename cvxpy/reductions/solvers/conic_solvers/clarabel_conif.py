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

This interface borrows heavily from the one in scs_conif.py
"""
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver


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

    # for dim in cone_dims.psd :
    #    PJG  : Placeholder for future PSD support

    for _ in range(cone_dims.exp):
        cones.append(clarabel.ExponentialConeT())

    for pow in cone_dims.p3d:
        cones.append(clarabel.PowerConeT(pow))
    return cones


class CLARABEL(ConicSolver):
    """An interface for the Clarabel solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS \
        + [SOC, ExpCone, PowCone3D]
    REQUIRES_CONSTR = True

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
        """Clarabel supports quadratic objective with any combination of conic constraints.
        """
        return True

    @staticmethod
    def extract_dual_value(result_vec, offset, constraint):
        """Extracts the dual value for constraint starting at offset.
        """

        # PJG : I will leave PSD handling from SCS here
        # as a placeholder to remind me to implement something
        # appropriate once PSD cones are supported

        if isinstance(constraint, PSD):
            raise RuntimeError("PSD cones not yet supported. This line should be unreachable.")

        else:
            return utilities.extract_dual_value(result_vec, offset,
                                                constraint)

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """

        attr = {}
        status = self.STATUS_MAP[str(solution.status)]
        attr[s.SOLVE_TIME] = solution.solve_time
        attr[s.NUM_ITERS] = solution.iterations
        # attr[s.EXTRA_STATS] = solution.extra.FOO #more detailed statistics here when available

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
    def parse_solver_opts(verbose, opts):
        import clarabel

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
            PJG: From SCS.   We don't support this, not sure if relevant
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
        c = data[s.C]

        if s.P in data:
            P = data[s.P]
        else:
            nvars = c.size
            P = sp.csc_matrix((nvars, nvars))

        cones = dims_to_solver_cones(data[ConicSolver.DIMS])

        def solve(_solver_opts):

            _settings = CLARABEL.parse_solver_opts(verbose, _solver_opts)
            _solver = clarabel.DefaultSolver(P, c, A, b, cones, _settings)
            _results = _solver.solve()

            return _results, _results.status

        results, status = solve(solver_opts)

        if solver_cache is not None and self.STATUS_MAP[str(status)]:
            solver_cache[self.name()] = results

        return results
