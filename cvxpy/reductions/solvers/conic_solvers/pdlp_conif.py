"""
Copyright, the CVXPY authors

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

import logging
from typing import Any, Dict, Tuple

import numpy as np
from scipy.sparse import csr_matrix

import cvxpy.settings as s
from cvxpy import Zero
from cvxpy.constraints import NonNeg
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.versioning import Version

log = logging.getLogger(__name__)


class PDLP(ConicSolver):
    """An interface to PDLP via OR-Tools."""

    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS

    # The key that maps to the pdlp.QuadraticProgram in the data returned by
    # apply().
    PDLP_MODEL = "pdlp_model"

    def name(self) -> str:
        """The name of the solver."""
        return 'PDLP'

    def import_solver(self) -> None:
        """Imports the solver."""
        import ortools  # noqa F401
        if Version(ortools.__version__) < Version('9.7.0'):
            raise RuntimeError(f'Version of ortools ({ortools.__version__}) '
                               f'is too old. Expected >= 9.7.0.')
        if Version(ortools.__version__) >= Version('9.12.0'):
            raise RuntimeError('Unrecognized new version of ortools '
                               f'({ortools.__version__}). Expected < 9.12.0. '
                               'Please open a feature request on cvxpy to '
                               'enable support for this version.')

    def apply(self, problem: ParamConeProg) -> Tuple[Dict, Dict]:
        """Returns a new problem and data for inverting the new solution."""
        from ortools.pdlp.python import pdlp

        # Create data and inv_data objects
        data = {}
        inv_data = {self.VAR_ID: problem.x.id}
        if not problem.formatted:
            problem = self.format_constraints(problem, None)
        data[s.PARAM_PROB] = problem
        data[self.DIMS] = problem.cone_dims

        constr_map = problem.constr_map
        inv_data["constraints"] = constr_map[Zero] + constr_map[NonNeg]

        # Min c'x + d such that Ax + b = s, s \in cones.
        c, d, A, b = problem.apply_parameters()
        A = csr_matrix(A)
        data["num_constraints"], data["num_vars"] = A.shape

        model = pdlp.QuadraticProgram()
        model.objective_offset = d.item() if isinstance(d, np.ndarray) else d
        model.objective_vector = c
        model.variable_lower_bounds = np.full_like(c, -np.inf)
        model.variable_upper_bounds = np.full_like(c, np.inf)

        model.constraint_matrix = A
        constraint_lower_bounds = np.full_like(b, -np.inf)
        constraint_upper_bounds = np.full_like(b, np.inf)
        # Ax + b = 0
        num_eq = problem.cone_dims.zero
        constraint_lower_bounds[:num_eq] = -b[:num_eq]
        constraint_upper_bounds[:num_eq] = -b[:num_eq]
        # Ax + b >= 0
        constraint_lower_bounds[num_eq:] = -b[num_eq:]

        model.constraint_lower_bounds = constraint_lower_bounds
        model.constraint_upper_bounds = constraint_upper_bounds

        data[self.PDLP_MODEL] = model
        return data, inv_data

    def invert(self, solution: Dict[str, Any],
               inverse_data: Dict[str, Any]) -> Solution:
        """Returns the solution to the original problem."""
        status = solution["status"]

        if status in s.SOLUTION_PRESENT:
            primal_vars = {
                inverse_data[self.VAR_ID]: solution["primal"]
            }
            dual_vars = utilities.get_dual_values(
                result_vec=solution["dual"],
                parse_func=utilities.extract_dual_value,
                constraints=inverse_data["constraints"],
            )
            return Solution(status, solution["value"], primal_vars, dual_vars,
                            {})
        else:
            return failure_solution(status)

    def solve_via_data(
            self,
            data: Dict[str, Any],
            warm_start: bool,
            verbose: bool,
            solver_opts: Dict[str, Any],
            solver_cache: Dict = None,
    ) -> Solution:
        """Returns the result of the call to the solver."""
        from ortools.pdlp import solvers_pb2
        from ortools.pdlp.python import pdlp

        parameters = solvers_pb2.PrimalDualHybridGradientParams()
        parameters.verbosity_level = 3 if verbose else 0
        # CVXPY reductions can leave a messy problem (e.g., no variable
        # bounds), so we turn on presolving by default.
        parameters.presolve_options.use_glop = True
        if "parameters_proto" in solver_opts:
            proto = solver_opts["parameters_proto"]
            if not isinstance(proto, solvers_pb2.PrimalDualHybridGradientParams):
                log.error("parameters_proto must be a PrimalDualHybridGradientParams")
                return {"status": s.SOLVER_ERROR}
            parameters.MergeFrom(proto)
        if "time_limit_sec" in solver_opts:
            limit = float(solver_opts["time_limit_sec"])
            parameters.termination_criteria.time_sec_limit = limit

        result = pdlp.primal_dual_hybrid_gradient(data[self.PDLP_MODEL],
                                                  parameters)
        solution = {}
        solution["primal"] = result.primal_solution
        solution["dual"] = result.dual_solution
        solution["status"] = self._status_map(result.solve_log)

        convergence_info = self._get_convergence_info(
            result.solve_log.solution_stats,
            result.solve_log.solution_type
        )
        if convergence_info is not None:
            solution["value"] = convergence_info.primal_objective
        else:
            solution["value"] = -np.inf

        return solution

    @staticmethod
    def _get_convergence_info(stats, candidate_type):
        for convergence_info in stats.convergence_information:
            if convergence_info.candidate_type == candidate_type:
                return convergence_info
        return None

    def _status_map(self, solve_log):
        from ortools.pdlp import solve_log_pb2
        TerminationReason = solve_log_pb2.TerminationReason
        status = solve_log.termination_reason
        if status == TerminationReason.TERMINATION_REASON_OPTIMAL:
            return s.OPTIMAL
        elif status == TerminationReason.TERMINATION_REASON_PRIMAL_INFEASIBLE:
            return s.INFEASIBLE
        elif status == TerminationReason.TERMINATION_REASON_DUAL_INFEASIBLE:
            # Not technically correct, but this seems to be the convention.
            return s.UNBOUNDED
        elif status == TerminationReason.TERMINATION_REASON_TIME_LIMIT:
            return s.USER_LIMIT
        elif status == TerminationReason.TERMINATION_REASON_ITERATION_LIMIT:
            return s.USER_LIMIT
        elif status == TerminationReason.TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT:
            return s.USER_LIMIT
        elif status == TerminationReason.TERMINATION_REASON_NUMERICAL_ERROR:
            log.warning('PDLP reported a numerical error.')
            return s.USER_LIMIT
        elif status == TerminationReason.TERMINATION_REASON_INVALID_PROBLEM:
            log.error('Invalid problem: %s', solve_log.termination_string)
            return s.SOLVER_ERROR
        elif status == TerminationReason.TERMINATION_REASON_INVALID_PARAMETER:
            log.error('Invalid parameter: %s', solve_log.termination_string)
            return s.SOLVER_ERROR
        elif status == TerminationReason.TERMINATION_REASON_PRIMAL_OR_DUAL_INFEASIBLE:
            return s.INFEASIBLE_OR_UNBOUNDED
        else:
            log.error("Unexpected status: %s Message: %s",
                      TerminationReason.Name(status),
                      solve_log.termination_string)
            return s.SOLVER_ERROR
