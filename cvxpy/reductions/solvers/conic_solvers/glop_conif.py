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

from numpy import array, ndarray
from scipy.sparse import csr_array

import cvxpy.settings as s
from cvxpy import Zero
from cvxpy.constraints import NonNeg
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.citations import CITATION_DICT
from cvxpy.utilities.versioning import Version

log = logging.getLogger(__name__)


class GLOP(ConicSolver):
    """An interface to Glop via OR-Tools."""

    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS

    # The key that maps to the MPModelProto in the data returned by apply().
    MODEL_PROTO = "model_proto"

    def name(self) -> str:
        """The name of the solver."""
        return 'GLOP'

    def import_solver(self) -> None:
        """Imports the solver."""
        import google.protobuf  # noqa F401
        import ortools  # noqa F401
        if Version(ortools.__version__) < Version('9.5.0'):
            raise RuntimeError(f'Version of ortools ({ortools.__version__}) '
                               f'is too old. Expected >= 9.5.0.')
        if Version(ortools.__version__) >= Version('9.15.0'):
            raise RuntimeError('Unrecognized new version of ortools '
                               f'({ortools.__version__}). Expected < 9.15.0. '
                               'Please open a feature request on cvxpy to '
                               'enable support for this version.')

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
        A = csr_array(A)
        data["num_constraints"], data["num_vars"] = A.shape

        # TODO: Switch to a vectorized model-building interface when one is
        # available in OR-Tools.
        model = linear_solver_pb2.MPModelProto()
        model.objective_offset = d.item() if isinstance(d, ndarray) else d
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
        from ortools.glop import parameters_pb2
        from ortools.linear_solver import linear_solver_pb2, pywraplp

        response = linear_solver_pb2.MPSolutionResponse()

        solver = pywraplp.Solver.CreateSolver('GLOP')
        solver.LoadModelFromProto(data[self.MODEL_PROTO])
        if verbose:
            solver.EnableOutput()
        if "parameters_proto" in solver_opts:
            proto = solver_opts["parameters_proto"]
            if not isinstance(proto, parameters_pb2.GlopParameters):
                log.error("parameters_proto must be a GlopParameters")
                return {"status": s.SOLVER_ERROR}
            proto_str = text_format.MessageToString(proto)
            if not solver.SetSolverSpecificParametersAsString(proto_str):
                return {"status": s.SOLVER_ERROR}
        if "time_limit_sec" in solver_opts:
            solver.SetTimeLimit(int(1000 * solver_opts["time_limit_sec"]))
        solver.Solve()
        solver.FillSolutionResponseProto(response)

        solution = {}
        solution["value"] = response.objective_value
        solution["status"] = self._status_map(response)
        has_primal = data["num_vars"] == 0 or len(response.variable_value) > 0
        if has_primal:
            solution["primal"] = array(response.variable_value)
        else:
            solution["primal"] = None
        has_dual = data["num_constraints"] == 0 or len(response.dual_value) > 0
        if has_dual:
            solution["dual"] = array(response.dual_value)
        else:
            solution["dual"] = None

        # Make solution status more precise depending on whether a solution is
        # present.
        if solution["status"] == s.SOLVER_ERROR and has_primal and has_dual:
            solution["status"] = s.USER_LIMIT

        return solution

    def _status_map(self, response):
        from ortools.linear_solver import linear_solver_pb2
        MPSolverResponseStatus = linear_solver_pb2.MPSolverResponseStatus
        status = response.status
        if status == MPSolverResponseStatus.MPSOLVER_OPTIMAL:
            return s.OPTIMAL
        elif status == MPSolverResponseStatus.MPSOLVER_FEASIBLE:
            return s.USER_LIMIT
        elif status == MPSolverResponseStatus.MPSOLVER_INFEASIBLE:
            return s.INFEASIBLE
        elif status == MPSolverResponseStatus.MPSOLVER_UNBOUNDED:
            return s.UNBOUNDED
        elif status == MPSolverResponseStatus.MPSOLVER_ABNORMAL:
            return s.SOLVER_ERROR
        # Skipping NOT_SOLVED and MODEL_IS_VALID because they shouldn't occur
        # for statuses obtained from a response.
        elif status == MPSolverResponseStatus.MPSOLVER_CANCELLED_BY_USER:
            return s.SOLVER_ERROR
        elif status == MPSolverResponseStatus.MPSOLVER_MODEL_INVALID:
            log.error("Solver reported that the model is invalid. Message: %s",
                      response.status_str)
            return s.SOLVER_ERROR
        # Skipping MPSOLVER_MODEL_INVALID_SOLUTION_HINT because we don't accept
        # solution hints.
        elif status == MPSolverResponseStatus.MPSOLVER_MODEL_INVALID_SOLVER_PARAMETERS:  # noqa
            log.error("Invalid solver parameters: %s", response.status_str)
            return s.SOLVER_ERROR
        else:
            log.warning("Unrecognized status: %s Message: %s",
                        linear_solver_pb2.MPSolverResponseStatus.Name(status),
                        response.status_str)
            return s.SOLVER_ERROR
    
    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["GLOP"]
