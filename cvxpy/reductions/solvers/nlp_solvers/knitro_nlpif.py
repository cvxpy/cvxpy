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


class KNITRO(NLPsolver):
    """
    NLP interface for the KNITRO solver
    """

    BOUNDED_VARIABLES = True
    # Keys:
    CONTEXT_KEY = "context"
    X_INIT_KEY = "x_init"
    Y_INIT_KEY = "y_init"

    # Keyword arguments for the CVXPY interface.
    INTERFACE_ARGS = [X_INIT_KEY, Y_INIT_KEY]

    # Map of Knitro status to CVXPY status.
    # This is based on the Knitro documentation:
    # https://www.artelys.com/app/docs/knitro/3_referenceManual/returnCodes.html
    STATUS_MAP = {
        0: s.OPTIMAL,
        -100: s.OPTIMAL_INACCURATE,
        -101: s.USER_LIMIT,
        -102: s.USER_LIMIT,
        -103: s.USER_LIMIT,
        -200: s.INFEASIBLE,
        -201: s.INFEASIBLE,
        -202: s.INFEASIBLE,
        -203: s.INFEASIBLE,
        -204: s.INFEASIBLE,
        -205: s.INFEASIBLE,
        -300: s.UNBOUNDED,
        -301: s.UNBOUNDED,
        -400: s.USER_LIMIT,
        -401: s.USER_LIMIT,
        -402: s.USER_LIMIT,
        -403: s.USER_LIMIT,
        -404: s.USER_LIMIT,
        -405: s.USER_LIMIT,
        -406: s.USER_LIMIT,
        -410: s.USER_LIMIT,
        -411: s.USER_LIMIT,
        -412: s.USER_LIMIT,
        -413: s.USER_LIMIT,
        -415: s.USER_LIMIT,
        -416: s.USER_LIMIT,
        -500: s.SOLVER_ERROR,
        -501: s.SOLVER_ERROR,
        -502: s.SOLVER_ERROR,
        -503: s.SOLVER_ERROR,
        -504: s.SOLVER_ERROR,
        -505: s.SOLVER_ERROR,
        -506: s.SOLVER_ERROR,
        -507: s.SOLVER_ERROR,
        -508: s.SOLVER_ERROR,
        -509: s.SOLVER_ERROR,
        -510: s.SOLVER_ERROR,
        -511: s.SOLVER_ERROR,
        -512: s.SOLVER_ERROR,
        -513: s.SOLVER_ERROR,
        -514: s.SOLVER_ERROR,
        -515: s.SOLVER_ERROR,
        -516: s.SOLVER_ERROR,
        -517: s.SOLVER_ERROR,
        -518: s.SOLVER_ERROR,
        -519: s.SOLVER_ERROR,
        -520: s.SOLVER_ERROR,
        -521: s.SOLVER_ERROR,
        -522: s.SOLVER_ERROR,
        -523: s.SOLVER_ERROR,
        -524: s.SOLVER_ERROR,
        -525: s.SOLVER_ERROR,
        -526: s.SOLVER_ERROR,
        -527: s.SOLVER_ERROR,
        -528: s.SOLVER_ERROR,
        -529: s.SOLVER_ERROR,
        -530: s.SOLVER_ERROR,
        -531: s.SOLVER_ERROR,
        -532: s.SOLVER_ERROR,
        -600: s.SOLVER_ERROR,
    }

    def name(self):
        """
        The name of solver.
        """
        return 'KNITRO'

    def import_solver(self):
        """
        Imports the solver.
        """
        import knitro  # noqa F401

    def invert(self, solution, inverse_data):
        """
        Returns the solution to the original problem given the inverse_data.
        """
        attr = {
            s.NUM_ITERS: solution.get('num_iters'),
            s.SOLVE_TIME: solution.get('solve_time_real'),
        }
        status = self.STATUS_MAP[solution['status']]
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
        Data used by the solver.
        This consists of:
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
            Additional arguments for the solver.
        solver_cache: None
            None

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        import knitro
        # Extract data from the data dictionary
        x0 = data["x0"]
        lb, ub = data["lb"].copy(), data["ub"].copy()
        cl, cu = data["cl"].copy(), data["cu"].copy()

        lb[lb == -np.inf] = -knitro.KN_INFINITY
        ub[ub == np.inf] = knitro.KN_INFINITY
        cl[cl == -np.inf] = -knitro.KN_INFINITY
        cu[cu == np.inf] = knitro.KN_INFINITY

        n = len(x0)  # number of variables
        m = len(cl)  # number of constraints

        # Create a new Knitro solver instance
        kc = knitro.KN_new()

        try:
            # Add variables
            knitro.KN_add_vars(kc, n)
            # Set variable bounds
            knitro.KN_set_var_lobnds(kc, xLoBnds=lb)
            knitro.KN_set_var_upbnds(kc, xUpBnds=ub)

            # Set initial values for variables
            knitro.KN_set_var_primal_init_values(kc, xInitVals=x0)

            # Add constraints (if any)
            if m > 0:
                knitro.KN_add_cons(kc, m)
                knitro.KN_set_con_lobnds(kc, cLoBnds=cl)
                knitro.KN_set_con_upbnds(kc, cUpBnds=cu)

            # Set objective goal to minimize
            knitro.KN_set_obj_goal(kc, knitro.KN_OBJGOAL_MINIMIZE)
            # Set verbosity
            if not verbose:
                knitro.KN_set_int_param(kc, knitro.KN_PARAM_OUTLEV, 0)

            # Get oracles for function evaluations
            oracles = data["oracles"]

            # Define the callback for evaluating objective and constraints (EVALFC)
            def callbackEvalFC(kc, cb, evalRequest, evalResult, userParams):
                if evalRequest.type != knitro.KN_RC_EVALFC:
                    return -1  # Error: wrong evaluation type

                # Convert x from list to numpy array
                x = np.array(evalRequest.x)

                # Evaluate objective
                evalResult.obj = oracles.objective(x)

                # Evaluate constraints (if any)
                if m > 0:
                    c_vals = oracles.constraints(x)
                    evalResult.c = c_vals
                return 0  # Success

            # Register the evaluation callback
            cb = knitro.KN_add_eval_callback(
                kc,
                evalObj=True,
                indexCons=(list(range(m)) if m > 0 else None),
                funcCallback=callbackEvalFC,
            )

            # Get the Jacobian sparsity structure
            if m > 0:
                jac_rows, jac_cols = oracles.jacobianstructure()
            else:
                jac_rows = None
                jac_cols = None

            # Define the callback for evaluating gradients (EVALGA)
            def callbackEvalGA(kc, cb, evalRequest, evalResult, userParams):
                if evalRequest.type != knitro.KN_RC_EVALGA:
                    return -1  # Error: wrong evaluation type

                try:
                    x = np.array(evalRequest.x)
                    # Evaluate objective gradient
                    grad = oracles.gradient(x)
                    evalResult.objGrad = np.asarray(grad).flatten()

                    # Evaluate constraint Jacobian (if any)
                    if m > 0:
                        jac_vals = oracles.jacobian(x)
                        evalResult.jac = np.asarray(jac_vals).flatten()

                    return 0  # Success
                except Exception as e:
                    print(f"Error in callbackEvalGA: {e}")
                    return -1

            # Register the gradient callback with sparsity structure
            knitro.KN_set_cb_grad(
                kc,
                cb,
                objGradIndexVars=list(range(n)),
                jacIndexCons=jac_rows,
                jacIndexVars=jac_cols,
                gradCallback=callbackEvalGA,
            )
            # oracles.hessianstructure() returns lower triangular (rows >= cols)
            # KNITRO expects upper triangular, so we swap rows and cols
            hess_cols, hess_rows = oracles.hessianstructure()

            # Define the callback for evaluating Hessian (EVALH)
            def callbackEvalH(kc, cb, evalRequest, evalResult, userParams):
                if evalRequest.type not in (knitro.KN_RC_EVALH, knitro.KN_RC_EVALH_NO_F):
                    return -1  # Error: wrong evaluation type

                try:
                    x = np.array(evalRequest.x)
                    # Get sigma (objective factor) and lambda (constraint multipliers)
                    sigma = evalRequest.sigma
                    lambda_ = np.array(evalRequest.lambda_)

                    # For KN_RC_EVALH_NO_F, the objective component should not be included
                    if evalRequest.type == knitro.KN_RC_EVALH_NO_F:
                        sigma = 0.0

                    # Evaluate Hessian of the Lagrangian
                    hess_vals = oracles.hessian(x, lambda_, sigma)
                    hess_vals = np.asarray(hess_vals).flatten()

                    evalResult.hess = hess_vals

                    return 0  # Success
                except Exception as e:
                    print(f"Error in callbackEvalH: {e}")
                    return -1

            # Register the Hessian callback with sparsity structure
            knitro.KN_set_cb_hess(
                kc,
                cb,
                hessIndexVars1=hess_rows,
                hessIndexVars2=hess_cols,
                hessCallback=callbackEvalH,
            )

            # Use exact Hessian by default (can be overridden by solver_opts)
            knitro.KN_set_int_param(kc, knitro.KN_PARAM_HESSOPT, knitro.KN_HESSOPT_EXACT)

            # Apply solver options from solver_opts
            # Map common string option names to KNITRO parameter constants
            OPTION_MAP = {
                'algorithm': knitro.KN_PARAM_ALGORITHM,
                'maxit': knitro.KN_PARAM_MAXIT,
                'outlev': knitro.KN_PARAM_OUTLEV,
                'hessopt': knitro.KN_PARAM_HESSOPT,
                'gradopt': knitro.KN_PARAM_GRADOPT,
                'feastol': knitro.KN_PARAM_FEASTOL,
                'opttol': knitro.KN_PARAM_OPTTOL,
                'honorbnds': knitro.KN_PARAM_HONORBNDS,
            }

            if solver_opts:
                for option_name, option_value in solver_opts.items():
                    # Map string names to KNITRO param IDs
                    if isinstance(option_name, str):
                        option_name_lower = option_name.lower()
                        if option_name_lower in OPTION_MAP:
                            param_id = OPTION_MAP[option_name_lower]
                        else:
                            raise ValueError(f"Unknown KNITRO option: {option_name}")
                    else:
                        # Assume it's already a KNITRO param ID
                        param_id = option_name

                    # Set the parameter based on value type
                    if isinstance(option_value, int):
                        knitro.KN_set_int_param(kc, param_id, option_value)
                    elif isinstance(option_value, float):
                        knitro.KN_set_double_param(kc, param_id, option_value)

            # Solve the problem
            nStatus = knitro.KN_solve(kc)

            # Retrieve the solution
            nStatus, objSol, x_sol, lambda_sol = knitro.KN_get_solution(kc)

            # Retrieve solve statistics
            num_iters = knitro.KN_get_number_iters(kc)
            solve_time_cpu = knitro.KN_get_solve_time_cpu(kc)
            solve_time_real = knitro.KN_get_solve_time_real(kc)

            # Return results in dictionary format expected by invert()
            solution = {
                'status': nStatus,
                'obj_val': objSol,
                'x': np.array(x_sol),
                'lambda': np.array(lambda_sol),
                'num_iters': num_iters,
                'solve_time_cpu': solve_time_cpu,
                'solve_time_real': solve_time_real,
            }
            return solution

        finally:
            # Always free the Knitro context
            knitro.KN_free(kc)

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["KNITRO"]
