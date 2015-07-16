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
from cvxpy.problems.solvers.ecos_intf import ECOS
import numpy as np

class SCS(ECOS):
    """An interface for the SCS solver.
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = True
    SDP_CAPABLE = True
    EXP_CAPABLE = True
    MIP_CAPABLE = False

    # Map of SCS status to CVXPY status.
    STATUS_MAP = {"Solved": s.OPTIMAL,
                  "Solved/Inaccurate": s.OPTIMAL_INACCURATE,
                  "Unbounded": s.UNBOUNDED,
                  "Unbounded/Inaccurate": s.UNBOUNDED_INACCURATE,
                  "Infeasible": s.INFEASIBLE,
                  "Infeasible/Inaccurate": s.INFEASIBLE_INACCURATE,
                  "Failure": s.SOLVER_ERROR,
                  "Indeterminate": s.SOLVER_ERROR}

    def name(self):
        """The name of the solver.
        """
        return s.SCS

    def import_solver(self):
        """Imports the solver.
        """
        import scs

    def split_constr(self, constr_map):
        """Extracts the equality, inequality, and nonlinear constraints.

        Parameters
        ----------
        constr_map : dict
            A dict of the canonicalized constraints.

        Returns
        -------
        tuple
            (eq_constr, ineq_constr, nonlin_constr)
        """
        return (constr_map[s.EQ] + constr_map[s.LEQ], [], [])

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
            Should the previous solver result be used to warm_start?
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        import scs
        data = self.get_problem_data(objective,
                                     constraints,
                                     cached_data)
        # Set the options to be VERBOSE plus any user-specific options.
        solver_opts["verbose"] = verbose
        scs_args = {"c": data[s.C], "A": data[s.A], "b": data[s.B]}
        # If warm_starting, add old primal and dual variables.
        solver_cache = cached_data[self.name()]
        if warm_start and solver_cache.prev_result is not None:
            scs_args["x"] = solver_cache.prev_result["x"]
            scs_args["y"] = solver_cache.prev_result["y"]
            scs_args["s"] = solver_cache.prev_result["s"]

        results_dict = scs.solve(scs_args, data[s.DIMS], **solver_opts)
        return self.format_results(results_dict, data, cached_data)

    def format_results(self, results_dict, data, cached_data):
        """Converts the solver output into standard form.

        Parameters
        ----------
        results_dict : dict
            The solver output.
        data : dict
            Information about the problem.
        cached_data : dict
            A map of solver name to cached problem data.

        Returns
        -------
        dict
            The solver output in standard form.
        """
        solver_cache = cached_data[self.name()]
        dims = data[s.DIMS]
        new_results = {}
        status = self.STATUS_MAP[results_dict["info"]["status"]]
        new_results[s.STATUS] = status
        if new_results[s.STATUS] in s.SOLUTION_PRESENT:
            # Save previous result for possible future warm_start.
            solver_cache.prev_result = {"x": results_dict["x"],
                                        "y": results_dict["y"],
                                        "s": results_dict["s"]}
            primal_val = results_dict["info"]["pobj"]
            new_results[s.VALUE] = primal_val + data[s.OFFSET]
            new_results[s.PRIMAL] = results_dict["x"]
            new_results[s.EQ_DUAL] = results_dict["y"][:dims[s.EQ_DIM]]
            y = results_dict["y"][dims[s.EQ_DIM]:]
            old_sdp_sizes = sum([n*(n+1)//2 for n in dims[s.SDP_DIM]])
            new_sdp_sizes = sum([n*n for n in dims[s.SDP_DIM]])
            y_true = np.zeros(y.shape[0] + (new_sdp_sizes - old_sdp_sizes))
            y_offset = dims[s.LEQ_DIM] + sum([n for n in dims[s.SOC_DIM]])
            y_true_offset = y_offset
            y_true[:y_true_offset] = y[:y_offset]
            # Expand SDP duals from lower triangular to full matrix,
            # scaling off diagonal entries by 1/sqrt(2).
            for n in dims[s.SDP_DIM]:
                tri = y[y_offset:y_offset+n*(n+1)//2]
                y_true[y_true_offset:y_true_offset+n*n] = self.tri_to_full(tri, n)
                y_true_offset += n*n
                y_offset += n*(n+1)//2
            y_true[y_true_offset:] = y[y_offset:]
            new_results[s.INEQ_DUAL] = y_true
        else:
            # No result to save.
            solver_cache.prev_result = None

        return new_results

    @staticmethod
    def tri_to_full(lower_tri, n):
        """Expands n*(n+1)//2 lower triangular to full matrix,
        with off-diagonal entries scaled by 1/sqrt(2).
        """
        full = np.zeros((n,n))
        for col in range(n):
            for row in range(col, n):
                idx = row - col + n*(n+1)//2 - (n-col)*(n-col+1)//2
                if row != col:
                    full[row, col] = lower_tri[idx]/np.sqrt(2)
                    full[col, row] = lower_tri[idx]/np.sqrt(2)
                else:
                    full[row, col] = lower_tri[idx]
        return np.reshape(full, n*n, order='F')
