"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import cvxpy.settings as s
from cvxpy.problems.solvers.cvxopt_intf import CVXOPT

class MOSEK(CVXOPT):
    """An interface for the MOSEK solver.
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = True
    SDP_CAPABLE = False
    EXP_CAPABLE = False
    MIP_CAPABLE = False

    def name(self):
        """The name of the solver.
        """
        return s.MOSEK

    def import_solver(self):
        """Imports the solver.
        """
        import mosek

    def solve(self, objective, constraints, cached_data,
              warm_start, verbose, solver_opts):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        objective : LinOp
            The canonicalized objective.
        constraints : list
            The list of canonicalized cosntraints.
        cached_data : dict
            A map of solver name to cached problem data.
        warm_start : bool
            Not used.
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        import vxopt.msk, mosek
        data = self.get_problem_data(objective, constraints, cached_data)
        # Save original cvxopt solver options.
        old_options = cvxopt.msk.options.copy()
        # Silence mosek if verbose is False.
        cvxopt.msk.options[mosek.iparam.log] = verbose

        # Apply any user-specific options.
        # Rename max_iters to maxiters.
        if "max_iters" in solver_opts:
            solver_opts["maxiters"] = solver_opts["max_iters"]
        for key, value in solver_opts.items():
            cvxopt.msk.options[key] = value

        try:
            results_dict = cvxopt.msk.conelp(data[s.C],
                                             data[s.G],
                                             data[s.H],
                                             data[s.A],
                                             data[s.B],
                                             solver="mosek")

         # Catch exceptions in CVXOPT and convert them to solver errors.
        except ValueError:
            results_dict = {"status": "unknown"}

        # Restore original cvxopt solver options.
        cvxopt.msk.options = old_options
        return self.format_results(results_dict, data, cached_data)
