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
from cvxpy.error import SolverError
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
    from cuopt.linear_programming.solver.solver_parameters import (
        CUOPT_ABSOLUTE_DUAL_TOLERANCE,
        CUOPT_ABSOLUTE_GAP_TOLERANCE,
        CUOPT_ABSOLUTE_PRIMAL_TOLERANCE,
        CUOPT_CROSSOVER,
        CUOPT_DUAL_INFEASIBLE_TOLERANCE,
        CUOPT_FIRST_PRIMAL_FEASIBLE,
        CUOPT_INFEASIBILITY_DETECTION,
        CUOPT_ITERATION_LIMIT,
        CUOPT_LOG_FILE,
        CUOPT_LOG_TO_CONSOLE,
        CUOPT_METHOD,
        CUOPT_MIP_ABSOLUTE_GAP,
        CUOPT_MIP_ABSOLUTE_TOLERANCE,
        CUOPT_MIP_HEURISTICS_ONLY,
        CUOPT_MIP_INTEGRALITY_TOLERANCE,
        CUOPT_MIP_RELATIVE_GAP,
        CUOPT_MIP_RELATIVE_TOLERANCE,
        CUOPT_MIP_SCALING,
        CUOPT_NUM_CPU_THREADS,
        CUOPT_PDLP_SOLVER_MODE,
        CUOPT_PER_CONSTRAINT_RESIDUAL,
        CUOPT_PRIMAL_INFEASIBLE_TOLERANCE,
        CUOPT_RELATIVE_DUAL_TOLERANCE,
        CUOPT_RELATIVE_GAP_TOLERANCE,
        CUOPT_RELATIVE_PRIMAL_TOLERANCE,
        CUOPT_SAVE_BEST_PRIMAL_SO_FAR,
        CUOPT_SOLUTION_FILE,
        CUOPT_STRICT_INFEASIBILITY,
        CUOPT_TIME_LIMIT,
        CUOPT_USER_PROBLEM_FILE,
    )
    from cuopt.linear_programming.solver.solver_wrapper import (
        ErrorStatus,
        LPTerminationStatus,
        MILPTerminationStatus,
    )
    from cuopt.linear_programming.solver_settings import (
        PDLPSolverMode,
        SolverMethod,
        SolverSettings,
    )

    cuopt_present = True
except Exception:
    cuopt_present = False

    from enum import IntEnum
    class MILPTerminationStatus(IntEnum):
        NoTermination = 0,
        Optimal = 1,
        FeasibleFound = 2,
        Infeasible = 3,
        Unbounded = 4,
        TimeLimit = 5

    class LPTerminationStatus(IntEnum):
        NoTermination = 0,
        NumericalError = 1,
        Optimal = 2,
        PrimalInfeasible = 3,
        DualInfeasible = 4,
        IterationLimit = 5,
        TimeLimit = 6,
        PrimalFeasible = 7

class CUOPT(ConicSolver):
    """ An interface to the cuOpt solver
    """
    # Solver capabilities.
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS
    MI_SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS
    BOUNDED_VARIABLES = True
    REQUIRES_CONSTR = True
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
        solver_modes = {"Stable2": PDLPSolverMode.Stable2,
                        "Methodical1": PDLPSolverMode.Methodical1,
                        "Fast1": PDLPSolverMode.Fast1}
        return solver_modes.get(m, PDLPSolverMode.Stable2)


    def _solver_method(self, m):
        solver_methods = {"Concurrent": SolverMethod.Concurrent,
                          "PDLP": SolverMethod.PDLP,
                          "DualSimplex": SolverMethod.DualSimplex}
        return solver_methods.get(m, SolverMethod.Concurrent)

    def name(self):
        """The name of the solver.
        """
        return s.CUOPT

    def import_solver(self) -> None:
        """Imports the solver.
        """
        if not cuopt_present:
            raise ModuleNotFoundError()

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data, inv_data = super(CUOPT, self).apply(problem)

        # Save the objective offset so that it can be set in the solver
        data[s.OFFSET] = inv_data[s.OFFSET]
        
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data['lp'] = not (data[s.BOOL_IDX] or data[s.INT_IDX])

        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution.status

        if status in s.SOLUTION_PRESENT:
            dual_vars = None
            opt_val = solution.opt_val + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: solution.primal_vars}
            if s.EQ_DUAL in solution.dual_vars and inverse_data['lp']:
                dual_vars = {}
                if len(inverse_data[self.EQ_CONSTR]) > 0:
                    eq_dual = utilities.get_dual_values(
                        solution.dual_vars[s.EQ_DUAL],
                        utilities.extract_dual_value,
                        inverse_data[self.EQ_CONSTR])
                    dual_vars.update(eq_dual)
                if len(inverse_data[self.NEQ_CONSTR]) > 0:
                    leq_dual = utilities.get_dual_values(
                        solution.dual_vars[s.INEQ_DUAL],
                        utilities.extract_dual_value,
                        inverse_data[self.NEQ_CONSTR])
                    dual_vars.update(leq_dual)

            return Solution(status, opt_val, primal_vars, dual_vars, solution.attr)
        else:
            return failure_solution(status)

    # Returns a SolverSettings object
    def _get_solver_settings(self, solver_opts, mip, verbose):
        ss = SolverSettings()
        ss.set_parameter(CUOPT_LOG_TO_CONSOLE, verbose)

        # Special handling for the enum value
        if CUOPT_PDLP_SOLVER_MODE in solver_opts:
            ss.set_parameter(CUOPT_PDLP_SOLVER_MODE,
                             self._solver_mode(solver_opts[CUOPT_PDLP_SOLVER_MODE]))

        # Name collision with "method" in cvxpy
        if "solver_method" in solver_opts:
            ss.set_parameter(CUOPT_METHOD, self._solver_method(solver_opts["solver_method"]))

        if "optimality" in solver_opts:
            ss.set_optimality_tolerance(solver_opts["optimality"])

        for p in [
                CUOPT_ABSOLUTE_DUAL_TOLERANCE,
                CUOPT_ABSOLUTE_GAP_TOLERANCE,
                CUOPT_ABSOLUTE_PRIMAL_TOLERANCE,
                CUOPT_CROSSOVER,
                CUOPT_DUAL_INFEASIBLE_TOLERANCE,
                CUOPT_FIRST_PRIMAL_FEASIBLE,
                CUOPT_INFEASIBILITY_DETECTION,
                CUOPT_ITERATION_LIMIT,
                CUOPT_LOG_FILE,
                CUOPT_MIP_ABSOLUTE_GAP,
                CUOPT_MIP_ABSOLUTE_TOLERANCE,
                CUOPT_MIP_HEURISTICS_ONLY,
                CUOPT_MIP_INTEGRALITY_TOLERANCE,
                CUOPT_MIP_RELATIVE_GAP,
                CUOPT_MIP_RELATIVE_TOLERANCE,
                CUOPT_MIP_SCALING,
                CUOPT_NUM_CPU_THREADS,
                CUOPT_PER_CONSTRAINT_RESIDUAL,
                CUOPT_PRIMAL_INFEASIBLE_TOLERANCE,
                CUOPT_RELATIVE_DUAL_TOLERANCE,
                CUOPT_RELATIVE_GAP_TOLERANCE,
                CUOPT_RELATIVE_PRIMAL_TOLERANCE,
                CUOPT_SAVE_BEST_PRIMAL_SO_FAR,
                CUOPT_SOLUTION_FILE,
                CUOPT_STRICT_INFEASIBILITY,
                CUOPT_TIME_LIMIT,
                CUOPT_USER_PROBLEM_FILE]:
            if p in solver_opts:
                ss.set_parameter(p, solver_opts[p])
        return ss


    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        verbose = verbose or solver_opts.get("solver_verbose", False) in [True,"True","true"]
        csr = data[s.A].tocsr(copy=False)
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
        variable_lower_bounds = data[s.LOWER_BOUNDS]
        variable_upper_bounds = data[s.UPPER_BOUNDS]
        if variable_lower_bounds is None:
            variable_lower_bounds = np.full(num_vars, -np.inf)
        if variable_upper_bounds is None:
            variable_upper_bounds = np.full(num_vars, np.inf)

        # Change bools to ints and set bounds
        is_mip = data[s.BOOL_IDX] or data[s.INT_IDX]
        if is_mip:
            # Set variable types
            variable_types[data[s.BOOL_IDX] + data[s.INT_IDX]] = 'I'

            # Make sure bounds for bool variables are [0,1]
            variable_lower_bounds[data[s.BOOL_IDX]] = 0
            variable_upper_bounds[data[s.BOOL_IDX]] = 1

        from cuopt.linear_programming.data_model import DataModel
        from cuopt.linear_programming.solver import Solve

        data_model = DataModel()
        data_model.set_csr_constraint_matrix(csr.data, csr.indices, csr.indptr)
        data_model.set_objective_coefficients(data['c'])
        data_model.set_objective_offset(data[s.OFFSET])
        data_model.set_constraint_lower_bounds(lower_bounds)
        data_model.set_constraint_upper_bounds(upper_bounds)

        data_model.set_variable_lower_bounds(variable_lower_bounds)
        data_model.set_variable_upper_bounds(variable_upper_bounds)
        data_model.set_variable_types(variable_types)

        ss = self._get_solver_settings(solver_opts, is_mip, verbose)
        cuopt_result = Solve(data_model, ss)

        if verbose:
            print('Termination reason: ', cuopt_result.get_termination_reason())
        if cuopt_result.get_error_status() != ErrorStatus.Success:
            raise SolverError(cuopt_result.get_error_message())

        dual_vars = {}
        if is_mip:
            sol_status = self.STATUS_MAP_MIP[cuopt_result.get_termination_status()]
            extra_stats = cuopt_result.get_milp_stats()
            iters = extra_stats["num_simplex_iterations"]
        else:
            d = cuopt_result.get_dual_solution()
            if d is not None:
                dual_vars[s.EQ_DUAL] = -d[0:leq_start]
                dual_vars[s.INEQ_DUAL] = -d[leq_start:leq_end]
            sol_status = self.STATUS_MAP_LP[cuopt_result.get_termination_status()]
            extra_stats = cuopt_result.get_lp_stats()
            iters = extra_stats["nb_iterations"]

        # Note, this is not the final solution. It is processed in invert()
        # and an updated Solution is returned
        solution = Solution(sol_status,
                            cuopt_result.get_primal_objective(),
                            cuopt_result.get_primal_solution(),
                            dual_vars,
                            attr={s.SOLVE_TIME: cuopt_result.get_solve_time(),
                                  s.NUM_ITERS: iters,
                                  s.EXTRA_STATS: extra_stats})

        return solution

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["CUOPT"]
