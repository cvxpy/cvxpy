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

    # The key that maps to the MPModelProto in the data returned by apply().
    MODEL_PROTO = "model_proto"

    def name(self) -> str:
        """The name of the solver."""
        return 'PDLP'

    def import_solver(self) -> None:
        """Imports the solver."""
        import google.protobuf
        import ortools
        if Version(ortools.__version__) < Version('9.3.0'):
            raise RuntimeError(f'Version of ortools ({ortools.__version__}) '
                               f'is too old. Expected >= 9.3.0.')
        ortools, google.protobuf  # For flake8

    def apply(self, problem: ParamConeProg) -> Tuple[Dict, Dict]:
        """Returns a new problem and data for inverting the new solution."""
        from ortools.linear_solver import linear_solver_pb2

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

        # TODO: Switch to a vectorized model-building interface when one is
        # available in OR-Tools.
        model = linear_solver_pb2.MPModelProto()
        model.objective_offset = d
        for var_index, obj_coef in enumerate(c):
            var = linear_solver_pb2.MPVariableProto(
                objective_coefficient=obj_coef,
                name="x_%d" % var_index)
            model.variable.append(var)

        for row_index in range(A.shape[0]):
            constraint = linear_solver_pb2.MPConstraintProto(
                name="constraint_%d" % row_index)
            start = A.indptr[row_index]
            end = A.indptr[row_index + 1]
            for nz_index in range(start, end):
                col_index = A.indices[nz_index]
                coeff = A.data[nz_index]
                constraint.var_index.append(col_index)
                constraint.coefficient.append(coeff)
            if row_index < problem.cone_dims.zero:
                # a'x + b == 0
                constraint.lower_bound = -b[row_index]
                constraint.upper_bound = -b[row_index]
            else:
                # a'x + b >= 0
                constraint.lower_bound = -b[row_index]
            model.constraint.append(constraint)

        data[self.MODEL_PROTO] = model
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
        from google.protobuf import text_format
        from ortools.linear_solver import linear_solver_pb2
        from ortools.model_builder.python import model_builder_helper
        from ortools.pdlp import solvers_pb2

        # TODO: Switch to a direct numpy interface to PDLP when available.
        # model_builder_helper is known to be slow because of proto
        # serialization.
        pdlp_solver = linear_solver_pb2.MPModelRequest.PDLP_LINEAR_PROGRAMMING
        request = linear_solver_pb2.MPModelRequest(
            model=data[self.MODEL_PROTO],
            enable_internal_solver_output=verbose,
            solver_type=pdlp_solver
        )
        parameters = solvers_pb2.PrimalDualHybridGradientParams()
        # CVXPY reductions can leave a messy problem (e.g., no variable bounds),
        # so we turn on presolving by default.
        parameters.presolve_options.use_glop = True
        if "parameters_proto" in solver_opts:
            proto = solver_opts["parameters_proto"]
            if not isinstance(proto, solvers_pb2.PrimalDualHybridGradientParams):
                log.error("parameters_proto must be a PrimalDualHybridGradientParams")
                return {"status": s.SOLVER_ERROR}
            parameters.MergeFrom(proto)
        if "time_limit_sec" in solver_opts:
            request.solver_time_limit_sec = float(solver_opts["time_limit_sec"])

        request.solver_specific_parameters = text_format.MessageToString(parameters)
        solver = model_builder_helper.ModelSolverHelper()
        response = solver.Solve(request)

        solution = {}
        solution["value"] = response.objective_value
        solution["status"] = self._status_map(response)
        solution["primal"] = np.array(response.variable_value)
        solution["dual"] = np.array(response.dual_value)

        return solution

    def _status_map(self, response):
        from ortools.pdlp import solve_log_pb2
        solve_log = solve_log_pb2.SolveLog.FromString(response.solver_specific_info)
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
