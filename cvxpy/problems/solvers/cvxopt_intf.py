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

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.problems.solvers.solver import Solver
from cvxpy.problems.kktsolver import get_kktsolver
import cvxopt
import cvxopt.solvers

class CVXOPT(Solver):
    """An interface for the CVXOPT solver.
    """
    def name(self):
        """The name of the solver.
        """
        return s.CVXOPT

    def matrix_intf(self):
        """The interface for matrices passed to the solver.
        """
        return intf.CVXOPT_SPARSE_INTF

    def vec_intf(self):
        """The interface for vectors passed to the solver.
        """
        return intf.CVXOPT_DENSE_INTF

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
        return (constr_map[s.EQ], constr_map[s.LEQ], constr_map[s.EXP])

    def _shape_args(self, c, A, b, G, h, F, dims):
        """Returns the arguments that will be passed to the solver.
        """
        if dims[s.EXP_DIM]:
            return (c, F, G, h, dims, A, b)
        else:
            return (c, G, h, dims, A, b)

    def solve(self, objective, constraints, cached_data, verbose, solver_opts):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        objective : LinOp
            The canonicalized objective.
        constraints : list
            The list of canonicalized cosntraints.
        cached_data : dict
            A map of solver name to cached problem data.
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        prob_data = self.get_problem_data(objective, constraints, cached_data)
        dims = prob_data[0][-3] # TODO this is a hack.
        obj_offset = prob_data[1]
        # Save original cvxopt solver options.
        old_options = cvxopt.solvers.options
        # Silence cvxopt if verbose is False.
        cvxopt.solvers.options['show_progress'] = verbose
        # Always do one step of iterative refinement after solving KKT system.
        cvxopt.solvers.options['refinement'] = 1

        # Apply any user-specific options.
        # Rename max_iters to maxiters.
        if "max_iters" in solver_opts:
            solver_opts["maxiters"] = solver_opts["max_iters"]
        for key, value in solver_opts.items():
            cvxopt.solvers.options[key] = value

        try:
            # Target cvxopt clp if nonlinear constraints exist
            if dims[s.EXP_DIM]:
                _, F, G, _, dims, A, _ = prob_data[0]
                # Get custom kktsolver.
                kktsolver = get_kktsolver(G, dims, A, F)
                results_dict = cvxopt.solvers.cpl(*prob_data[0],
                                                  kktsolver=kktsolver)
            else:
                _, G, _, dims, A, _ = prob_data[0]
                # Get custom kktsolver.
                kktsolver = get_kktsolver(G, dims, A)
                results_dict = cvxopt.solvers.conelp(*prob_data[0],
                                                     kktsolver=kktsolver)
        # Catch exceptions in CVXOPT and convert them to solver errors.
        except ValueError:
            results_dict = {'status': 'unknown'}

        # Restore original cvxopt solver options.
        cvxopt.solvers.options = old_options
        return self.format_results(results_dict, dims, obj_offset)

    def format_results(self, results_dict, dims, obj_offset=0):
        """Converts the solver output into standard form.

        Parameters
        ----------
        results_dict : dict
            The solver output.
        dims : dict
            The cone dimensions in the canonicalized problem.
        obj_offset : float, optional
            The constant term in the objective.

        Returns
        -------
        dict
            The solver output in standard form.
        """
        new_results = {}
        status = s.SOLVER_STATUS[s.CVXOPT][results_dict['status']]
        new_results[s.STATUS] = status
        if new_results[s.STATUS] in s.SOLUTION_PRESENT:
            primal_val = results_dict['primal objective']
            new_results[s.VALUE] = primal_val + obj_offset
            new_results[s.PRIMAL] = results_dict['x']
            new_results[s.EQ_DUAL] = results_dict['y']
            if dims[s.EXP_DIM]:
                new_results[s.INEQ_DUAL] = results_dict['zl']
            else:
                new_results[s.INEQ_DUAL] = results_dict['z']

        return new_results
