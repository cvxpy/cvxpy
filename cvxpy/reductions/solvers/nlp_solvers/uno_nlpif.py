"""
Copyright 2025, the CVXPY developers

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
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import NLPsolver
from cvxpy.utilities.citations import CITATION_DICT


class UNO(NLPsolver):
    """
    NLP interface for the Uno solver.

    Uno is a modern nonlinear optimization solver that unifies Lagrange-Newton
    methods by decomposing them into modular building blocks for constraint
    relaxation, descent directions, and globalization strategies.

    For more information, see: https://github.com/cvanaret/Uno
    """

    STATUS_MAP = {
        # Success cases (optimization_status)
        "SUCCESS": s.OPTIMAL,

        # Limit cases
        "ITERATION_LIMIT": s.USER_LIMIT,
        "TIME_LIMIT": s.USER_LIMIT,

        # Error cases
        "EVALUATION_ERROR": s.SOLVER_ERROR,
        "ALGORITHMIC_ERROR": s.SOLVER_ERROR,

        # Solution status cases
        "FEASIBLE_KKT_POINT": s.OPTIMAL,
        "FEASIBLE_FJ_POINT": s.OPTIMAL_INACCURATE,
        "FEASIBLE_SMALL_STEP": s.OPTIMAL_INACCURATE,
        "INFEASIBLE_STATIONARY_POINT": s.INFEASIBLE,
        "INFEASIBLE_SMALL_STEP": s.INFEASIBLE,
        "UNBOUNDED": s.UNBOUNDED,
        "NOT_OPTIMAL": s.SOLVER_ERROR,
    }

    def name(self):
        """
        The name of solver.
        """
        return 'UNO'

    def import_solver(self):
        """
        Imports the solver.
        """
        import unopy  # noqa F401

    def invert(self, solution, inverse_data):
        """
        Returns the solution to the original problem given the inverse_data.
        """
        attr = {}

        # Get status from the solution - try optimization_status first,
        # then solution_status
        status_key = solution.get('optimization_status', solution.get('solution_status'))
        status = self.STATUS_MAP.get(str(status_key), s.SOLVER_ERROR)

        attr[s.NUM_ITERS] = solution.get('iterations', 0)
        if 'cpu_time' in solution:
            attr[s.SOLVE_TIME] = solution['cpu_time']

        if status in s.SOLUTION_PRESENT:
            primal_val = solution['obj_val']
            opt_val = primal_val + inverse_data.offset
            primal_vars = {}
            x_opt = solution['x']
            for id, offset in inverse_data.var_offsets.items():
                shape = inverse_data.var_shapes[id]
                size = np.prod(shape, dtype=int)
                primal_vars[id] = np.reshape(x_opt[offset:offset+size], shape, order='F')
            return Solution(status, opt_val, primal_vars, {}, attr)
        else:
            return failure_solution(status, attr)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """
        Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data used by the solver. This consists of:
            - "oracles": An Oracles object that computes the objective and constraints
            - "x0": Initial guess for the primal variables
            - "lb": Lower bounds on the primal variables
            - "ub": Upper bounds on the primal variables
            - "cl": Lower bounds on the constraints
            - "cu": Upper bounds on the constraints
        warm_start : bool
            Not used.
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver. Common options include:
            - "preset": Solver preset ("filtersqp" or "ipopt")
            - Any other Uno option name-value pairs
        solver_cache: None
            Not used.

        Returns
        -------
        dict
            Solution dictionary with status, objective value, and primal solution.
        """
        import unopy

        from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import Oracles

        # Create oracles object (deferred from apply() so we have access to verbose)
        bounds = data["_bounds"]

        # UNO uses the exact Hessian, or reverts to L-BFGS when no Hessian is provided
        use_hessian = True

        if solver_cache is None:
            oracles = Oracles(bounds.new_problem, verbose=verbose, use_hessian=use_hessian)
        elif 'oracles' in solver_cache:
            oracles = solver_cache['oracles']
            if bounds.new_problem.parameters():
                oracles.update_params(bounds.new_problem)
        else:
            oracles = Oracles(bounds.new_problem, verbose=verbose, use_hessian=use_hessian)
            solver_cache['oracles'] = oracles

        # Extract data from the data dictionary
        x0 = data["x0"]
        lb = data["lb"].copy()
        ub = data["ub"].copy()
        cl = data["cl"].copy()
        cu = data["cu"].copy()

        n = len(x0)  # number of variables
        m = len(cl)  # number of constraints

        # Create the Uno model using unopy constants
        # Model(problem_type, number_variables, variables_lower_bounds,
        #       variables_upper_bounds, base_indexing)
        model = unopy.Model(
            unopy.PROBLEM_NONLINEAR,
            n,
            lb,
            ub,
            unopy.ZERO_BASED_INDEXING
        )

        # Define callbacks. unopy catches possible exceptions

        # Define objective function callback
        # Signature: objective(x) -> double
        # Must return result
        def objective_callback(x):
            return oracles.objective(x)

        # Define objective gradient callback
        # Signature: gradient(x, gradient) -> void
        # Must write result to gradient array
        def gradient_callback(x, gradient):
            gradient[:] = oracles.gradient(x)

        # Set objective (minimization)
        model.set_objective(unopy.MINIMIZE, objective_callback, gradient_callback)

        # Set constraints if there are any
        if m > 0:
            # Define constraints callback
            # Signature: constraints(x, constraint_values) -> void
            def constraints_callback(x, constraint_values):
                constraint_values[:] = oracles.constraints(x)

            # Get Jacobian sparsity structure
            jac_rows, jac_cols = oracles.jacobianstructure()
            nnz_jacobian = len(jac_rows)

            # Define Jacobian callback
            # Signature: jacobian(x, jacobian_values) -> void
            def jacobian_callback(x, jacobian_values):
                jacobian_values[:] = oracles.jacobian(x)

            # set_constraints(number_constraints, constraint_functions,
            #     constraints_lower_bounds, constraints_upper_bounds,
            #     number_jacobian_nonzeros, jacobian_row_indices,
            #     jacobian_column_indices, constraint_jacobian)
            model.set_constraints(
                m,
                constraints_callback,
                cl,
                cu,
                nnz_jacobian,
                jac_rows,
                jac_cols,
                jacobian_callback
            )

        # Get Hessian sparsity structure
        # oracles.hessianstructure() returns lower triangular (rows >= cols)
        hess_rows, hess_cols = oracles.hessianstructure()
        nnz_hessian = len(hess_rows)

        # Define Lagrangian Hessian callback
        # Signature: hessian(x, objective_multiplier, multipliers, hessian_values)
        def hessian_callback(x, objective_multiplier, multipliers, hessian_values):
            hessian_values[:] = oracles.hessian(x, multipliers, objective_multiplier)

        # set_lagrangian_hessian(number_hessian_nonzeros, hessian_triangular_part,
        #     hessian_row_indices, hessian_column_indices, lagrangian_hessian)
        # hessian_triangular_part: LOWER_TRIANGLE since we store lower triangular
        model.set_lagrangian_hessian(
            nnz_hessian,
            unopy.LOWER_TRIANGLE,
            hess_rows,
            hess_cols,
            hessian_callback
        )
        # set_lagrangian_sign_convention(lagrangian_sign_convention)
        # lagrangian_sign_convention: MULTIPLIER_POSITIVE means L = sigma*f + lambda*g
        # This matches our oracles.hessian convention
        model.set_lagrangian_sign_convention(unopy.MULTIPLIER_POSITIVE)

        # Set initial primal iterate
        model.set_initial_primal_iterate(x0)

        # Create solver and configure
        uno_solver = unopy.UnoSolver()

        # Make a copy of solver_opts to avoid modifying the original
        opts = dict(solver_opts) if solver_opts else {}

        # Set default preset (can be overridden by solver_opts)
        default_preset = opts.pop("preset", "filtersqp")
        uno_solver.set_preset(default_preset)

        # Set verbosity
        if not verbose:
            uno_solver.set_option("print_solution", False)

        # Apply user-provided solver options
        for option_name, option_value in opts.items():
            uno_solver.set_option(option_name, option_value)

        # Solve the problem
        result = uno_solver.optimize(model)

        # Extract solution information
        # Convert enum to string for status mapping
        opt_status = str(result.optimization_status).split('.')[-1]
        sol_status = str(result.solution_status).split('.')[-1]

        solution = {
            'optimization_status': opt_status,
            'solution_status': sol_status,
            'obj_val': result.solution_objective,
            'x': result.primal_solution,
            'iterations': result.number_iterations,
            'cpu_time': result.cpu_time,
            'primal_feasibility': result.solution_primal_feasibility,
            'stationarity': result.solution_stationarity,
            'complementarity': result.solution_complementarity,
        }

        # Include dual solutions if available
        if hasattr(result, 'constraint_dual_solution'):
            solution['constraint_dual'] = np.array(list(result.constraint_dual_solution))
        if hasattr(result, 'lower_bound_dual_solution'):
            solution['lower_bound_dual'] = np.array(list(result.lower_bound_dual_solution))
        if hasattr(result, 'upper_bound_dual_solution'):
            solution['upper_bound_dual'] = np.array(list(result.upper_bound_dual_solution))

        return solution

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["UNO"]
