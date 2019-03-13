"""
Copyright 2013 Steven Diamond

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

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.reductions.solvers.conic_solvers.cvxopt_conif import CVXOPT
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solution import Solution, failure_solution


class GLPK(CVXOPT):
    """An interface for the GLPK solver.
    """
    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS

    def name(self):
        """The name of the solver.
        """
        return s.GLPK

    def import_solver(self):
        """Imports the solver.
        """
        from cvxopt import glpk
        glpk  # For flake8

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data, inv_data = super(CVXOPT, self).apply(problem)
        # Convert A, b, G, h, c to CVXOPT matrices.
        if data[s.A] is not None:
            data[s.A] = intf.sparse2cvxopt(data[s.A])
        if data[s.G] is not None:
            data[s.G] = intf.sparse2cvxopt(data[s.G])
        if data[s.B] is not None:
            data[s.B] = intf.dense2cvxopt(data[s.B])
        if data[s.H] is not None:
            data[s.H] = intf.dense2cvxopt(data[s.H])
        if data[s.C] is not None:
            data[s.C] = intf.dense2cvxopt(data[s.C])
        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']

        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value'] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: solution['primal']}
            return Solution(status, opt_val, primal_vars, dual_vars, {})
        else:
            return failure_solution(status)

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        import cvxopt
        import cvxopt.solvers
        # Save original cvxopt solver options.
        old_options = cvxopt.solvers.options.copy()
        # Silence cvxopt if verbose is False.
        if verbose:
            cvxopt.solvers.options["msg_lev"] = "GLP_MSG_ON"
        else:
            cvxopt.solvers.options["msg_lev"] = "GLP_MSG_OFF"

        # Apply any user-specific options.
        # Rename max_iters to maxiters.
        if "max_iters" in solver_opts:
            solver_opts["maxiters"] = solver_opts["max_iters"]
        for key, value in solver_opts.items():
            cvxopt.solvers.options[key] = value

        try:
            results_dict = cvxopt.solvers.lp(data[s.C],
                                             data[s.G],
                                             data[s.H],
                                             data[s.A],
                                             data[s.B],
                                             solver="glpk")

        # Catch exceptions in CVXOPT and convert them to solver errors.
        except ValueError:
            results_dict = {"status": "unknown"}

        # Restore original cvxopt solver options.
        self._restore_solver_options(old_options)

        # Convert CVXOPT results to solution format.
        solution = {}
        status = self.STATUS_MAP[results_dict['status']]
        solution[s.STATUS] = status
        if solution[s.STATUS] in s.SOLUTION_PRESENT:
            primal_val = results_dict['primal objective']
            solution[s.VALUE] = primal_val
            solution[s.PRIMAL] = results_dict['x']
            solution[s.EQ_DUAL] = results_dict['y']
            solution[s.INEQ_DUAL] = results_dict['z']
            for key in [s.PRIMAL, s.EQ_DUAL, s.INEQ_DUAL]:
                solution[key] = intf.cvxopt2dense(solution[key])

        return solution

    @staticmethod
    def _restore_solver_options(old_options):
        import cvxopt.solvers
        for key, value in list(cvxopt.solvers.options.items()):
            if key in old_options:
                cvxopt.solvers.options[key] = old_options[key]
            else:
                del cvxopt.solvers.options[key]
