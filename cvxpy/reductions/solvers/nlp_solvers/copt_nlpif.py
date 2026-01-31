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


class COPT(NLPsolver):
    """
    NLP interface for the COPT solver.
    """
    # Map between COPT status and CVXPY status
    STATUS_MAP = {
                  1: s.OPTIMAL,             # optimal
                  2: s.INFEASIBLE,          # infeasible
                  3: s.UNBOUNDED,           # unbounded
                  4: s.INF_OR_UNB,          # infeasible or unbounded
                  5: s.SOLVER_ERROR,        # numerical
                  6: s.USER_LIMIT,          # node limit
                  7: s.OPTIMAL_INACCURATE,  # imprecise
                  8: s.USER_LIMIT,          # time out
                  9: s.SOLVER_ERROR,        # unfinished
                  10: s.USER_LIMIT          # interrupted
                 }

    def name(self):
        """
        The name of solver.
        """
        return 'COPT'

    def import_solver(self):
        """
        Imports the solver.
        """
        import coptpy  # noqa F401

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
        - "objective": Function to compute the objective value
        - "gradient": Function to compute the objective gradient
        - "constraints": Function to compute the constraint values
        - "jacobian": Function to compute the constraint Jacobian
        - "jacobianstructure": Function to compute the structure of the Jacobian
        - "hessian": Function to compute the Hessian of the Lagrangian
        - "hessianstructure": Function to compute the structure of the Hessian
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
        import coptpy as copt

        from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import Oracles

        # Create oracles object (deferred from apply() so we have access to verbose)
        bounds = data["_bounds"]
        oracles = Oracles(bounds.new_problem, bounds.x0, len(bounds.cl), verbose=verbose)

        class COPTNlpCallbackCVXPY(copt.NlpCallbackBase):
            def __init__(self, oracles, m):
                super().__init__()
                self._oracles = oracles
                self._m = m

            def EvalObj(self, xdata, outdata):
                x = copt.NdArray(xdata)
                outval = copt.NdArray(outdata)

                x_np = x.tonumpy()
                outval_np = self._oracles.objective(x_np)

                outval[:] = outval_np
                return 0

            def EvalGrad(self, xdata, outdata):
                x = copt.NdArray(xdata)
                outval = copt.NdArray(outdata)

                x_np = x.tonumpy()
                outval_np = self._oracles.gradient(x_np)

                outval[:] = np.asarray(outval_np).flatten()
                return 0

            def EvalCon(self, xdata, outdata):
                if self._m > 0:
                    x = copt.NdArray(xdata)
                    outval = copt.NdArray(outdata)

                    x_np = x.tonumpy()
                    outval_np = self._oracles.constraints(x_np)

                    outval[:] = np.asarray(outval_np).flatten()
                return 0

            def EvalJac(self, xdata, outdata):
                if self._m > 0:
                    x = copt.NdArray(xdata)
                    outval = copt.NdArray(outdata)

                    x_np = x.tonumpy()
                    outval_np = self._oracles.jacobian(x_np)

                    outval[:] = np.asarray(outval_np).flatten()
                return 0

            def EvalHess(self, xdata, sigma, lambdata, outdata):
                x = copt.NdArray(xdata)
                lagrange = copt.NdArray(lambdata)
                outval = copt.NdArray(outdata)

                x_np = x.tonumpy()
                lagrange_np = lagrange.tonumpy()
                outval_np = self._oracles.hessian(x_np, lagrange_np, sigma)

                outval[:] = np.asarray(outval_np).flatten()
                return 0

        # Create COPT environment and model
        envconfig = copt.EnvrConfig()
        if not verbose:
            envconfig.set('nobanner', '1')

        env = copt.Envr(envconfig)
        model = env.createModel()

        # Pass through verbosity
        model.setParam(copt.COPT.Param.Logging, verbose)

        # Get the NLP problem data
        x0 = data['x0']
        lb, ub = data['lb'].copy(), data['ub'].copy()
        cl, cu = data['cl'].copy(), data['cu'].copy()

        lb[lb == -np.inf] = -copt.COPT.INFINITY
        ub[ub == +np.inf] = +copt.COPT.INFINITY
        cl[cl == -np.inf] = -copt.COPT.INFINITY
        cu[cu == +np.inf] = +copt.COPT.INFINITY

        n = len(lb)
        m = len(cl)

        cbtype = copt.COPT.EVALTYPE_OBJVAL | copt.COPT.EVALTYPE_CONSTRVAL | \
                 copt.COPT.EVALTYPE_GRADIENT | copt.COPT.EVALTYPE_JACOBIAN | \
                 copt.COPT.EVALTYPE_HESSIAN
        cbfunc = COPTNlpCallbackCVXPY(oracles, m)

        if m > 0:
            jac_rows, jac_cols = oracles.jacobianstructure()
            nnz_jac = len(jac_rows)
        else:
            jac_rows = None
            jac_cols = None
            nnz_jac = 0

        if n > 0:
            hess_rows, hess_cols = oracles.hessianstructure()
            nnz_hess = len(hess_rows)
        else:
            hess_rows = None
            hess_cols = None
            nnz_hess = 0

        # Load NLP problem data
        model.loadNlData(n,                                  # Number of variables
                         m,                                  # Number of constraints
                         copt.COPT.MINIMIZE,                 # Objective sense
                         copt.COPT.DENSETYPE_ROWMAJOR, None, # Dense objective gradient
                         nnz_jac, jac_rows, jac_cols,        # Sparse jacobian
                         nnz_hess, hess_rows, hess_cols,     # Sparse hessian
                         lb, ub,                             # Variable bounds
                         cl, cu,                             # Constraint bounds
                         x0,                                 # Starting point
                         cbtype, cbfunc                      # Callback function
                         )

        # Set parameters
        for key, value in solver_opts.items():
            model.setParam(key, value)

        # Solve problem 
        model.solve()

        # Get solution
        nlp_status = model.status
        nlp_hassol = model.haslpsol

        if nlp_hassol:
            objval = model.objval
            x_sol = model.getValues()
            lambda_sol = model.getDuals()
        else:
            objval = +np.inf
            x_sol = [0.0] * n
            lambda_sol = [0.0] * m

        num_iters = model.barrieriter
        solve_time_real = model.solvingtime

        # Return results in dictionary format expected by invert()
        solution = {
            'status': nlp_status,
            'obj_val': objval,
            'x': np.array(x_sol),
            'lambda': np.array(lambda_sol),
            'num_iters': num_iters,
            'solve_time_real': solve_time_real
        }

        return solution

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["COPT"]
