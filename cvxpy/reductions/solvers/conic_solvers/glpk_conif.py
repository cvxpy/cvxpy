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
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.conic_solvers.cvxopt_conif import CVXOPT


class GLPK(CVXOPT):
    """An interface for the GLPK solver.
    """
    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS

    # When constraints aren't provided, use zero matrix and vector of this length
    MIN_CONSTRAINT_LENGTH = 1

    def name(self):
        """The name of the solver.
        """
        return s.GLPK

    def import_solver(self) -> None:
        """Imports the solver.
        """
        from cvxopt import glpk  # noqa F401

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        return super(GLPK, self).invert(solution, inverse_data)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        import cvxopt.solvers

        # Save original cvxopt solver options.
        old_options = cvxopt.solvers.options.copy()
        # Silence cvxopt if verbose is False.
        if verbose:
            cvxopt.solvers.options["msg_lev"] = "GLP_MSG_ON"
        else:
            if "glpk" in solver_opts:
                solver_opts["glpk"]["msg_lev"] = "GLP_MSG_OFF"
            else:
                solver_opts["glpk"] = {"msg_lev": "GLP_MSG_OFF"}

        data = self._prepare_cvxopt_matrices(data)

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
    def _restore_solver_options(old_options) -> None:
        import cvxopt.solvers
        for key, value in list(cvxopt.solvers.options.items()):
            if key in old_options:
                cvxopt.solvers.options[key] = old_options[key]
            else:
                del cvxopt.solvers.options[key]
