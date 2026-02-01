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

        # UNO always uses exact Hessian (no quasi-Newton option currently)
        use_hessian = True

        oracles = Oracles(bounds.new_problem, bounds.x0, len(bounds.cl),
                          verbose=verbose, use_hessian=use_hessian)

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
            lb.tolist(),
            ub.tolist(),
            unopy.ZERO_BASED_INDEXING
        )

        # Helper to convert unopy.Vector to numpy array
        # unopy.Vector doesn't support len() but supports indexing
        def to_numpy(vec, size):
            return np.array([vec[i] for i in range(size)])

        # Define objective function callback
        # Signature: objective(number_variables, x, objective_value, user_data) -> int
        # Must write result to objective_value[0] and return 0 on success
        def objective_callback(number_variables, x, objective_value, user_data):
            try:
                x_arr = to_numpy(x, number_variables)
                objective_value[0] = oracles.objective(x_arr)
                return 0
            except Exception:
                return 1

        # Define objective gradient callback
        # Signature: gradient(number_variables, x, gradient, user_data) -> int
        # Must write result to gradient array and return 0 on success
        def gradient_callback(number_variables, x, gradient, user_data):
            try:
                x_arr = to_numpy(x, number_variables)
                grad = oracles.gradient(x_arr)
                for i in range(n):
                    gradient[i] = grad[i]
                return 0
            except Exception:
                return 1

        # Set objective (minimization)
        model.set_objective(unopy.MINIMIZE, objective_callback, gradient_callback)

        # Set constraints if there are any
        if m > 0:
            # Define constraints callback
            # Signature: constraints(n, m, x, constraint_values, user_data) -> int
            def constraints_callback(number_variables, number_constraints, x,
                                     constraint_values, user_data):
                try:
                    x_arr = to_numpy(x, number_variables)
                    cons = oracles.constraints(x_arr)
                    for i in range(m):
                        constraint_values[i] = cons[i]
                    return 0
                except Exception:
                    return 1

            # Get Jacobian sparsity structure
            jac_rows, jac_cols = oracles.jacobianstructure()
            nnz_jacobian = len(jac_rows)

            # Define Jacobian callback
            # Signature: jacobian(n, nnz, x, jacobian_values, user_data) -> int
            def jacobian_callback(number_variables, number_jacobian_nonzeros, x,
                                  jacobian_values, user_data):
                try:
                    x_arr = to_numpy(x, number_variables)
                    jac_vals = oracles.jacobian(x_arr)
                    # Flatten in case it's returned as a 2D memoryview
                    jac_vals_arr = np.asarray(jac_vals).flatten()
                    for i in range(nnz_jacobian):
                        jacobian_values[i] = float(jac_vals_arr[i])
                    return 0
                except Exception:
                    return 1

            # set_constraints(number_constraints, constraint_functions,
            #     constraints_lower_bounds, constraints_upper_bounds,
            #     number_jacobian_nonzeros, jacobian_row_indices,
            #     jacobian_column_indices, constraint_jacobian)
            model.set_constraints(
                m,
                constraints_callback,
                cl.tolist(),
                cu.tolist(),
                nnz_jacobian,
                jac_rows.tolist(),
                jac_cols.tolist(),
                jacobian_callback
            )

        # Get Hessian sparsity structure
        # oracles.hessianstructure() returns lower triangular (rows >= cols)
        hess_rows, hess_cols = oracles.hessianstructure()
        nnz_hessian = len(hess_rows)

        # Define Lagrangian Hessian callback
        # Signature: hessian(n, m, nnz, x, obj_factor, multipliers, hessian_values, user_data)
        # Uno's MULTIPLIER_POSITIVE convention: L = sigma*f + sum_i lambda_i * g_i
        # This matches our oracles.hessian convention
        def hessian_callback(number_variables, number_constraints, number_hessian_nonzeros,
                             x, objective_multiplier, multipliers, hessian_values, user_data):
            try:
                x_arr = to_numpy(x, number_variables)
                mult_arr = to_numpy(multipliers, number_constraints) if m > 0 else np.array([])
                hess_vals = oracles.hessian(x_arr, mult_arr, objective_multiplier)
                # Flatten in case it's returned as a 2D array
                hess_vals_arr = np.asarray(hess_vals).flatten()
                for i in range(nnz_hessian):
                    hessian_values[i] = float(hess_vals_arr[i])
                return 0
            except Exception:
                return 1

        # set_lagrangian_hessian(number_hessian_nonzeros, hessian_triangular_part,
        #     hessian_row_indices, hessian_column_indices, lagrangian_hessian,
        #     lagrangian_sign_convention)
        # hessian_triangular_part: LOWER_TRIANGLE since we store lower triangular
        # lagrangian_sign_convention: MULTIPLIER_POSITIVE means L = sigma*f + lambda*g
        model.set_lagrangian_hessian(
            nnz_hessian,
            unopy.LOWER_TRIANGLE,
            hess_rows.tolist(),
            hess_cols.tolist(),
            hessian_callback,
            unopy.MULTIPLIER_POSITIVE
        )

        # Set initial primal iterate
        model.set_initial_primal_iterate(x0.tolist())

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
            uno_solver.set_option("statistics_print_header_frequency", 0)

        # Apply user-provided solver options
        for option_name, option_value in opts.items():
            uno_solver.set_option(option_name, option_value)

        # Solve the problem
        result = uno_solver.optimize(model)

        # Extract solution information
        # Convert enum to string for status mapping
        opt_status = str(result.optimization_status).split('.')[-1]
        sol_status = str(result.solution_status).split('.')[-1]

        # Convert unopy.Vector to numpy array via list()
        # (np.array(unopy.Vector) returns a 0-d object array, not what we want)
        solution = {
            'optimization_status': opt_status,
            'solution_status': sol_status,
            'obj_val': result.solution_objective,
            'x': np.array(list(result.primal_solution)),
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
