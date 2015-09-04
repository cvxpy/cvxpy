"""
Copyright 2015 Enzo Busseti

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
import numpy as np
import scipy.sparse as sp

class MOSEK(Solver):
    """An interface for the MOSEK solver.
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = True
    SDP_CAPABLE = False # for now only SOCP
    EXP_CAPABLE = False
    MIP_CAPABLE = False

    def import_solver(self):
        """Imports the solver.
        """
        import mosek

    def name(self):
        """The name of the solver.
        """
        return s.MOSEK

    def matrix_intf(self):
        """The interface for matrices passed to the solver.
        """
        return intf.DEFAULT_SPARSE_INTF

    def vec_intf(self):
        """The interface for vectors passed to the solver.
        """
        return intf.DEFAULT_INTF

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
        return (constr_map[s.EQ], constr_map[s.LEQ], [])

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
        import mosek
        env = mosek.Env()
        task = env.Task(0,0)

        if verbose:
            # Define a stream printer to grab output from MOSEK
            def streamprinter(text):
                import sys
                sys.stdout.write(text)
                sys.stdout.flush()
            env.set_Stream(mosek.streamtype.log, streamprinter)
            task.set_Stream(mosek.streamtype.log, streamprinter)

        data = self.get_problem_data(objective, constraints, cached_data)

        A = data[s.A]
        b = data[s.B]
        G = data[s.G]
        h = data[s.H]
        c = data[s.C]
        dims = data[s.DIMS]

        # size of problem
        numvar = len(c) + sum(dims[s.SOC_DIM])
        numcon = len(b) + dims[s.LEQ_DIM] + sum(dims[s.SOC_DIM])

        # objective
        task.appendvars(numvar)
        task.putclist(np.arange(len(c)), c)
        task.putvarboundlist(np.arange(numvar, dtype=int),
                             [mosek.boundkey.fr]*numvar,
                             np.zeros(numvar),
                             np.zeros(numvar))

        # linear equality and linear inequality constraints
        task.appendcons(numcon)
        if A.shape[0] and G.shape[0]:
            constraints_matrix = sp.bmat([[A], [G]])
        else:
            constraints_matrix = A if A.shape[0] else G
        coefficients = np.concatenate([b, h])

        row,col,el = sp.find(constraints_matrix)
        task.putaijlist(row,col,el)

        type_constraint = [mosek.boundkey.fx] * len(b)
        type_constraint += [mosek.boundkey.up] * dims[s.LEQ_DIM]
        type_constraint += [mosek.boundkey.fx] * (sum(dims[s.SOC_DIM]))

        task.putconboundlist(np.arange(numcon, dtype=int),
                             type_constraint,
                             coefficients,
                             coefficients)

        # cones
        current_var_index = len(c)
        current_con_index = len(b) + dims[s.LEQ_DIM]

        for size_cone in dims['q']:
            row,col,el = sp.find(sp.eye(size_cone))
            row += current_con_index
            col += current_var_index
            task.putaijlist(row,col,el) # add a identity for each cone
            # add a cone constraint
            task.appendcone(mosek.conetype.quad,
                            0.0, #unused
                            np.arange(current_var_index,
                                      current_var_index + size_cone))
            current_con_index += size_cone
            current_var_index += size_cone

        # solve
        task.putobjsense(mosek.objsense.minimize)
        task.optimize()

        if verbose:
            task.solutionsummary(mosek.streamtype.msg)

        return self.format_results(task, data, cached_data)

    def format_results(self, task, data, cached_data):
        """Converts the solver output into standard form.

        Parameters
        ----------
        task : mosek.Task
            The solver status interface.
        data : dict
            Information about the problem.
        cached_data : dict
            A map of solver name to cached problem data.

        Returns
        -------
        dict
            The solver output in standard form.
        """

        import mosek
        # Map of MOSEK status to CVXPY status.
        # taken from:
        # http://docs.mosek.com/7.0/pythonapi/Solution_status_keys.html
        STATUS_MAP = {mosek.solsta.optimal: s.OPTIMAL,
                      mosek.solsta.prim_infeas_cer: s.INFEASIBLE,
                      mosek.solsta.dual_infeas_cer: s.UNBOUNDED,
                      mosek.solsta.near_optimal: s.OPTIMAL_INACCURATE,
                      mosek.solsta.near_prim_infeas_cer: s.INFEASIBLE_INACCURATE,
                      mosek.solsta.near_dual_infeas_cer: s.UNBOUNDED_INACCURATE,
                      mosek.solsta.unknown: s.SOLVER_ERROR}

        prosta = task.getprosta(mosek.soltype.itr) #unused
        solsta = task.getsolsta(mosek.soltype.itr)

        result_dict = {s.STATUS: STATUS_MAP[solsta]}

        if result_dict[s.STATUS] in s.SOLUTION_PRESENT:
            # get primal variables values
            result_dict[s.PRIMAL] = np.zeros(task.getnumvar(), dtype=np.float)
            task.getxx(mosek.soltype.itr, result_dict[s.PRIMAL])
            # get obj value
            result_dict[s.VALUE] = task.getprimalobj(mosek.soltype.itr) + \
                                   data[s.OFFSET]
            # get dual
            y = np.zeros(task.getnumcon(), dtype=np.float)
            task.gety(mosek.soltype.itr, y)
            # it appears signs are inverted
            result_dict[s.EQ_DUAL] = -y[:len(data[s.B])]
            result_dict[s.INEQ_DUAL] = \
                -y[len(data[s.B]):len(data[s.B])+data[s.DIMS][s.LEQ_DIM]]

        return result_dict
