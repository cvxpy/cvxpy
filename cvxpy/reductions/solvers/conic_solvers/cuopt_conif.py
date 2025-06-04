"""
Copyright 2025 NVIDIA CORPORATION

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

import cvxpy.settings as s
from cvxpy.constraints import NonPos
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    ConicSolver,
    dims_to_solver_dict,
)
from cvxpy.utilities.citations import CITATION_DICT

# Wrap cuopt imports in an exception handler so that we
# can have them at module level but not break if cuoopt
# is not installed
try:
    from cuopt.linear_programming.solver.solver_wrapper import (
        MILPTerminationStatus,
        LPTerminationStatus,
        ErrorStatus
    )
    from cuopt.linear_programming.solver_settings import PDLPSolverMode
    from cuopt.linear_programming.solver_settings import SolverSettings
    from cuopt.linear_programming.solver.solver_parameters import (
        CUOPT_ABSOLUTE_DUAL_TOLERANCE,
        CUOPT_ABSOLUTE_GAP_TOLERANCE,
        CUOPT_ABSOLUTE_PRIMAL_TOLERANCE,
        CUOPT_CROSSOVER,
        CUOPT_DUAL_INFEASIBLE_TOLERANCE,
        CUOPT_INFEASIBILITY_DETECTION,
        CUOPT_ITERATION_LIMIT,
        CUOPT_LOG_TO_CONSOLE,
        CUOPT_METHOD,
        CUOPT_MIP_ABSOLUTE_GAP,
        CUOPT_MIP_HEURISTICS_ONLY,
        CUOPT_MIP_INTEGRALITY_TOLERANCE,
        CUOPT_MIP_RELATIVE_GAP,
        CUOPT_MIP_SCALING,
        CUOPT_NUM_CPU_THREADS,
        CUOPT_PDLP_SOLVER_MODE,
        CUOPT_PRIMAL_INFEASIBLE_TOLERANCE,
        CUOPT_RELATIVE_DUAL_TOLERANCE,
        CUOPT_RELATIVE_GAP_TOLERANCE,
        CUOPT_RELATIVE_PRIMAL_TOLERANCE,
        CUOPT_TIME_LIMIT,
        )
    cuopt_present = True
except Exception:
    cuopt_present = False

try:
    from cuopt_sh_client import CuOptServiceSelfHostClient
    cuopt_client_present = True
except Exception:
    cuopt_client_present = False

class CUOPT(ConicSolver):
    """ An interface to the cuOpt solver
    """
    # Solver capabilities.
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [NonPos]
    MI_SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS

    STATUS_MAP_MIP = {
        MILPTerminationStatus.NoTermination: s.SOLVER_ERROR,
        MILPTerminationStatus.Optimal: s.OPTIMAL,
        MILPTerminationStatus.FeasibleFound: s.USER_LIMIT,
        MILPTerminationStatus.Infeasible: s.INFEASIBLE,
        MILPTerminationStatus.Unbounded: s.UNBOUNDED,
        MILPTerminationStatus.TimeLimit: s.USER_LIMIT
    }

    STATUS_MAP_LP = {
        LPTerminationStatus.NoTermination: s.SOLVER_ERROR,
        LPTerminationStatus.NumericalError: s.SOLVER_ERROR,
        LPTerminationStatus.Optimal: s.OPTIMAL,
        LPTerminationStatus.PrimalInfeasible: s.INFEASIBLE,
        LPTerminationStatus.DualInfeasible: s.UNBOUNDED,
        LPTerminationStatus.IterationLimit: s.USER_LIMIT,
        LPTerminationStatus.TimeLimit: s.USER_LIMIT,
        LPTerminationStatus.PrimalFeasible: s.USER_LIMIT
    }

    def _solver_mode(self, m):
        solver_modes = {"Stable1": PDLPSolverMode.Stable1,
                        "Stable2": PDLPSolverMode.Stable2,
                        "Methodical1": PDLPSolverMode.Methodical1,
                        "Fast1": PDLPSolverMode.Fast1}
        return solver_modes[m]


    def name(self):
        """The name of the solver.
        """
        return s.CUOPT

    def import_solver(self) -> None:
        """Imports the solver.
        """
        self.local_install = cuopt_present
        self.service_install = cuopt_client_present
        if not (self.local_install or self.service_install):
            raise

    def accepts(self, problem) -> bool:
        """Can cuopt solve the problem?
        """
        # TODO check if is matrix stuffed.
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in CUOPT.SUPPORTED_CONSTRAINTS:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data, inv_data = super(CUOPT, self).apply(problem)
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data['lp'] = not (data[s.BOOL_IDX] or data[s.INT_IDX])

        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']

        if status in s.SOLUTION_PRESENT:
            dual_vars = None
            opt_val = solution['value'] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: solution['primal']}
            if s.EQ_DUAL in solution and inverse_data['lp']:
                dual_vars = {}
                if len(inverse_data[self.EQ_CONSTR]) > 0:
                    #print('solution[s.EQ_DUAL] ', solution[s.EQ_DUAL])
                    eq_dual = utilities.get_dual_values(
                        solution[s.EQ_DUAL],
                        utilities.extract_dual_value,
                        inverse_data[self.EQ_CONSTR])
                    dual_vars.update(eq_dual)
                if len(inverse_data[self.NEQ_CONSTR]) > 0:
                    #print('leq')
                    #print('solution[s.INEQ_DUAL] ', solution[s.INEQ_DUAL])
                    #print('inverse_data[self.NEQ_CONSTR] ', inverse_data[self.NEQ_CONSTR])
                    leq_dual = utilities.get_dual_values(
                        solution[s.INEQ_DUAL],
                        utilities.extract_dual_value,
                        inverse_data[self.NEQ_CONSTR])
                    dual_vars.update(leq_dual)


            return Solution(status, opt_val, primal_vars, dual_vars, {})
        else:
            return failure_solution(status)

    # Returns a SolverSettings object
    def _get_solver_settings(self, solver_opts, mip, verbose):
        ss = SolverSettings()
        # Always need to map to verbose
        ss.set_parameter(CUOPT_LOG_TO_CONSOLE, verbose)

        # Special handling for the enum value
        if CUOPT_PDLP_SOLVER_MODE in solver_opts:
            ss.set_parameter(CUOPT_PDLP_SOLVER_MODE,
                             self._solver_mode(solver_opts[CUOPT_PDLP_SOLVER_MODE]))

        # Name collision with "method" in cvxpy
        if "solver_method" in solver_opts:
            ss.set_parameter(CUOPT_METHOD, solver_opts["solver_method"])

        if "optimality" in solver_opts:
            ss.set_optimality_tolerance(solver_opts["optimality"])

        for p in [
                CUOPT_ABSOLUTE_DUAL_TOLERANCE,
                CUOPT_ABSOLUTE_GAP_TOLERANCE,
                CUOPT_ABSOLUTE_PRIMAL_TOLERANCE,
                CUOPT_CROSSOVER,
                CUOPT_DUAL_INFEASIBLE_TOLERANCE,
                CUOPT_INFEASIBILITY_DETECTION,
                CUOPT_ITERATION_LIMIT,
                CUOPT_MIP_ABSOLUTE_GAP,
                CUOPT_MIP_HEURISTICS_ONLY,
                CUOPT_MIP_INTEGRALITY_TOLERANCE,
                CUOPT_MIP_RELATIVE_GAP,
                CUOPT_MIP_SCALING,
                CUOPT_NUM_CPU_THREADS,
                CUOPT_PRIMAL_INFEASIBLE_TOLERANCE,
                CUOPT_RELATIVE_DUAL_TOLERANCE,
                CUOPT_RELATIVE_GAP_TOLERANCE,
                CUOPT_RELATIVE_PRIMAL_TOLERANCE,
                CUOPT_TIME_LIMIT]:
            if p in solver_opts:
                ss.set_parameter(p, solver_opts[p])
        return ss

    # Returns a dictionary
    def _get_solver_config(self, solver_opts, mip, verbose):

        def _apply(name, sc, alias=None):
            if name in solver_opts:
                if alias is None:
                    alias = name
                sc[alias] = solver_opts[name]

        solver_config = {}

        # Always need to map to verbose
        solver_config[CUOPT_LOG_TO_CONSOLE] = verbose

        # Special handling for the enum value
        if CUOPT_PDLP_SOLVER_MODE in solver_opts:
            solver_config[CUOPT_PDLP_SOLVER_MODE] = self._solver_mode(solver_opts[CUOPT_PDLP_SOLVER_MODE])

        # Name collision with "method" in cvxpy
        if "solver_method" in solver_opts:
            solver_config[CUOPT_METHOD] = solver_opts["solver_method"]

        for p in [
                CUOPT_CROSSOVER,
                CUOPT_INFEASIBILITY_DETECTION,
                CUOPT_ITERATION_LIMIT,
                CUOPT_MIP_HEURISTICS_ONLY,
                CUOPT_MIP_SCALING,
                CUOPT_NUM_CPU_THREADS,
                CUOPT_TIME_LIMIT]:
            _apply(p, solver_config)

        t = {}
        for name, alias in [
                (CUOPT_ABSOLUTE_DUAL_TOLERANCE,     "absolue_dual"),
                (CUOPT_ABSOLUTE_GAP_TOLERANCE,      "absolute_gap"),
                (CUOPT_ABSOLUTE_PRIMAL_TOLERANCE,   "absolute_primal"),
                (CUOPT_DUAL_INFEASIBLE_TOLERANCE,   "dual_infeasible"),
                (CUOPT_PRIMAL_INFEASIBLE_TOLERANCE, "primal_infeasible"),
                (CUOPT_RELATIVE_DUAL_TOLERANCE,     "relative_dual"),
                (CUOPT_RELATIVE_GAP_TOLERANCE,      "relative_gap"),
                (CUOPT_RELATIVE_PRIMAL_TOLERANCE,   "relative_primal"),
                (CUOPT_MIP_ABSOLUTE_GAP, None),
                (CUOPT_MIP_INTEGRALITY_TOLERANCE, None),
                (CUOPT_MIP_RELATIVE_GAP, None),
                ("optimality", None)]:
            _apply(name, t, alias)

        solver_config["tolerances"] = t
        return solver_config

    def _get_client(self, solver_opts):
        import requests

        # Do a health check based on the service arguments
        ip = solver_opts.get("service_host", "localhost")
        port = solver_opts.get("service_port", 5000)
        scheme = solver_opts.get("service_scheme", "http")
        try:
            loc = f"{scheme}://{ip}:{port}"
            requests.get(f"{loc}/coupt/health")
        except Exception:
            print("Error: cuopt service client is installed but cannot "
                  f"connect to the service at {loc}")
            raise
        return CuOptServiceSelfHostClient(ip=ip, port=port)


    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        use_service = solver_opts.get("use_service", False) in [True,"True","true"]
        if self.local_install ^ self.service_install:
            if self.local_install:
                if use_service:
                    print("Warning: use_service ignored since cuopt service is not available")
                use_service = False
            else:
                if not use_service:
                    print("Warning: use_service ignored since cuopt is not installed locally")
                use_service = True

        # Using copy=False here would be more efficient, but is anything on the calling side
        # using data[s.A] after this call?  Or is it okay to change it?
        csr = data[s.A].tocsr()
        #csr = data[s.A].tocsr(copy=False)

        num_vars = data['c'].shape[0]

        dims = dims_to_solver_dict(data[s.DIMS])
        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]

        # Get constraint bounds
        lower_bounds = np.empty(leq_end)
        lower_bounds[:leq_start] = data['b'][:leq_start]
        lower_bounds[leq_start:leq_end] = float('-inf')
        upper_bounds = data['b'][:leq_end].copy()

        # Initialize variable types and bounds
        variable_types = np.full(num_vars, 'C', dtype='U1')
        variable_lower_bounds = np.full(num_vars, -np.inf)
        variable_upper_bounds = np.full(num_vars, np.inf)

        # Change bools to ints and set bounds
        is_mip = data[s.BOOL_IDX] or data[s.INT_IDX]
        if is_mip:
            # Set variable types
            variable_types[data[s.BOOL_IDX] + data[s.INT_IDX]] = 'I'

            # Set bounds for bool variables to [0,1]
            variable_lower_bounds[data[s.BOOL_IDX]] = 0
            variable_upper_bounds[data[s.BOOL_IDX]] = 1

        # Now if we have variable bounds in solver_opts, optionally overwrite lower or upper
        if "variable_bounds" in solver_opts:
            vbounds = solver_opts["variable_bounds"]
            if "lower" in vbounds:
                variable_lower_bounds = vbounds["lower"]
            if "upper" in vbounds:
                variable_upper_bounds = vbounds["upper"]

        if use_service:
            d = {}
            d["maximize"] = False
            d["csr_constraint_matrix"] = {
                "offsets": csr.indptr.tolist(),
                "indices": csr.indices.tolist(),
                "values": csr.data.tolist()
            }
            d["objective_data"] = {
                "coefficients": data['c'].tolist(),
                "scalability_factor": 1,
                "offset": 0
            }
            d["variable_bounds"] = {
                "upper_bounds": variable_upper_bounds.tolist(),
                "lower_bounds": variable_lower_bounds.tolist()
            }
            d["constraint_bounds"] = {
                "upper_bounds": upper_bounds.tolist(),
                "lower_bounds": lower_bounds.tolist()
            }
            d["variable_types"] = variable_types.tolist()
            d["solver_config"] = self._get_solver_config(solver_opts, is_mip, verbose)

            cuopt_service_client = self._get_client(solver_opts)

            # In error case the client will raise an exception here
            res = cuopt_service_client.get_LP_solve(
                d, response_type='obj')["response"]["solver_response"]
            cuopt_result = res["solution"]

            # If conversion to an object didn't work, then this means that
            # we got an infeasible response or similar where expected fields were missing.
            # Since we only need a subset of the object, build it here.
            if isinstance(cuopt_result, dict):
                from cuopt.linear_programming.solution import Solution
                if is_mip:
                    pt = 1
                    dual_solution = None
                else:
                    pt = 0
                    dual_solution = cuopt_result.get("dual_solution", None)
                    if dual_solution:
                        dual_solution = np.array(dual_solution)

                primal_solution = cuopt_result.get("primal_solution", None)
                if primal_solution:
                    primal_solution = np.array(primal_solution)
                primal_objective = cuopt_result.get("primal_objective", 0.0)

                cuopt_result = Solution(problem_category=pt,
                                        vars=None,
                                        dual_solution=dual_solution,
                                        primal_solution=primal_solution,
                                        primal_objective=primal_objective,
                                        termination_status=res["status"])

        else:
            from cuopt.linear_programming.data_model import DataModel
            from cuopt.linear_programming.solver import Solve

            data_model = DataModel()
            data_model.set_csr_constraint_matrix(csr.data, csr.indices, csr.indptr)
            data_model.set_objective_coefficients(data['c'])
            data_model.set_constraint_lower_bounds(lower_bounds)
            data_model.set_constraint_upper_bounds(upper_bounds)

            data_model.set_variable_lower_bounds(variable_lower_bounds)
            data_model.set_variable_upper_bounds(variable_upper_bounds)
            data_model.set_variable_types(variable_types)

            ss = self._get_solver_settings(solver_opts, is_mip, verbose)
            cuopt_result = Solve(data_model, ss)


        print('Termination reason: ', cuopt_result.get_termination_reason())
        if cuopt_result.get_error_status() != ErrorStatus.Success:
            raise ValueError(cuopt_result.get_error_message())

        solution = {}
        if is_mip:
            solution["status"] = self.STATUS_MAP_MIP[cuopt_result.get_termination_status()]
        else:
            # This really ought to be a getter but the service version of this class is missing it
            # So just grab the result.
            d = cuopt_result.dual_solution
            if d is not None:
                solution[s.EQ_DUAL] = -d[0:leq_start]
                solution[s.INEQ_DUAL] = -d[leq_start:leq_end]
            solution["status"] = self.STATUS_MAP_LP[cuopt_result.get_termination_status()]

        solution["primal"] = cuopt_result.get_primal_solution()
        solution["value"] = cuopt_result.get_primal_objective()
        return solution

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["CUOPT"]
