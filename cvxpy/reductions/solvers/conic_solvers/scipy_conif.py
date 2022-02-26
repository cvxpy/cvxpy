"""
Copyright 2021 The CVXPY Developers

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


import warnings

import scipy  # For version checks

import cvxpy.settings as s
from cvxpy.constraints import NonNeg, Zero
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.versioning import Version


class SCIPY(ConicSolver):
    """An interface for the SciPy linprog function.
    Note: This requires a version of SciPy which is >= 1.6.1
    """

    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS

    # Map of SciPy linprog status
    STATUS_MAP = {0: s.OPTIMAL,  # Optimal
                  1: s.SOLVER_ERROR,  # Iteration limit reached
                  2: s.INFEASIBLE,  # Infeasible
                  3: s.UNBOUNDED,  # Unbounded
                  4: s.SOLVER_ERROR  # Numerical difficulties encountered
                  }

    def import_solver(self) -> None:
        """Imports the solver.
        """
        from scipy import optimize as opt
        opt  # For flake8

    def name(self):
        """The name of the solver.
        """
        return s.SCIPY

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data = {}
        inv_data = {self.VAR_ID: problem.x.id}

        if not problem.formatted:
            problem = self.format_constraints(problem, None)
        data[s.PARAM_PROB] = problem
        data[self.DIMS] = problem.cone_dims
        inv_data[self.DIMS] = problem.cone_dims

        constr_map = problem.constr_map
        inv_data[self.EQ_CONSTR] = constr_map[Zero]
        inv_data[self.NEQ_CONSTR] = constr_map[NonNeg]
        len_eq = problem.cone_dims.zero

        c, d, A, b = problem.apply_parameters()
        data[s.C] = c
        inv_data[s.OFFSET] = d
        data[s.A] = -A[:len_eq]
        if data[s.A].shape[0] == 0:
            data[s.A] = None
        data[s.B] = b[:len_eq].flatten()
        if data[s.B].shape[0] == 0:
            data[s.B] = None
        data[s.G] = -A[len_eq:]
        if 0 in data[s.G].shape:
            data[s.G] = None
        data[s.H] = b[len_eq:].flatten()
        if 0 in data[s.H].shape:
            data[s.H] = None
        return data, inv_data

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        from scipy import optimize as opt

        # Set default method which can be overriden by user inputs
        if (Version(scipy.__version__) < Version('1.6.1')):
            meth = "interior-point"
        else:
            meth = "highs"

        # Extract solver options which are not part of the options dictionary
        if solver_opts:

            # Raise error message if the parameters are not passed in
            # a dictionary called 'scipy_options'.
            if "scipy_options" not in solver_opts:
                raise ValueError("All parameters for the SCIPY solver should "
                                 "be incased within a dictionary called "
                                 "scipy_options e.g. \n"
                                 "prob.solve(solver='SCIPY', verbose=True,"
                                 " scipy_options={'method':'highs-ds', 'maxiter':10000})")

            # Raise warning if the 'method' parameter is not specified
            if "method" not in solver_opts['scipy_options']:
                warnings.warn("It is best to specify the 'method' parameter "
                              "within scipy_options. The main advantage "
                              "of this solver, is its ability to use the "
                              "HiGHS LP solvers via scipy.optimize.linprog() "
                              "which require a SciPy version >= 1.6.1 ."
                              "\n\nThe default method '{}' will be"
                              " used in this case.\n".format(meth))

            else:
                meth = solver_opts['scipy_options'].pop("method")

                # Check to see if scipy version larger than 1.6.1 is installed
                # if method chosen is one of the highs methods.
                ver = (Version(scipy.__version__) < Version('1.6.1'))
                if ((meth in ['highs-ds', 'highs-ipm', 'highs']) & ver):
                    raise ValueError("The HiGHS solvers require a SciPy version >= 1.6.1")

            # Disable the 'bounds' parameter to avoid problems with
            # canonicalised problems.
            if "bounds" in solver_opts['scipy_options']:
                raise ValueError("Please do not specify bounds through "
                                 "scipy_options. Please specify bounds "
                                 "through CVXPY.")

            # Not supported by HiGHS solvers:
            # callback = solver_opts['scipy_options'].pop("callback", None)
            # x0 = solver_opts['scipy_options'].pop("x0", None)

            # Run the optimisation using scipy.optimize.linprog
            solution = opt.linprog(data[s.C], A_ub=data[s.G], b_ub=data[s.H],
                                   A_eq=data[s.A], b_eq=data[s.B], method=meth,
                                   bounds=(None, None), options=solver_opts['scipy_options'])
        else:

            warnings.warn("It is best to specify the 'method' parameter "
                          "within scipy_options. The main advantage "
                          "of this solver, is its ability to use the "
                          "HiGHS LP solvers via scipy.optimize.linprog() "
                          "which require a SciPy version >= 1.6.1 ."
                          "\n\nThe default method '{}' will be"
                          " used in this case.\n".format(meth))

            # Run the optimisation using scipy.optimize.linprog
            solution = opt.linprog(data[s.C], A_ub=data[s.G], b_ub=data[s.H],
                                   A_eq=data[s.A], b_eq=data[s.B], method=meth,
                                   bounds=(None, None))

        if verbose is True:
            print("Solver terminated with message: " + solution.message)

        return solution

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = self.STATUS_MAP[solution['status']]

        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            primal_val = solution['fun']
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: solution['x']}

            # SciPy linprog only returns duals for version >= 1.7.0
            # and method is one of 'highs', 'highs-ds' or 'highs-ipm'
            if ('ineqlin' in solution.keys()):
                eq_dual = utilities.get_dual_values(
                    -solution['eqlin']['marginals'],
                    utilities.extract_dual_value,
                    inverse_data[self.EQ_CONSTR])
                leq_dual = utilities.get_dual_values(
                    -solution['ineqlin']['marginals'],
                    utilities.extract_dual_value,
                    inverse_data[self.NEQ_CONSTR])
                eq_dual.update(leq_dual)
                dual_vars = eq_dual

            return Solution(status, opt_val, primal_vars, dual_vars, {})
        else:
            return failure_solution(status)
