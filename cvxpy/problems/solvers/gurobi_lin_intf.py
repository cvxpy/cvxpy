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
import numpy as np
from cvxpy.problems.solvers.solver import Solver
from scipy.sparse import dok_matrix

class GUROBI_LIN(Solver):
    """An interface for the Gurobi solver.
    """

    # Solver capabilities.
    LP_CAPABLE = True

    # NOTE: Gurobi should be able to solve these kinds of problems
    # I just haven't tested them.
    SOCP_CAPABLE = False    
    SDP_CAPABLE = False
    EXP_CAPABLE = False
    MIP_CAPABLE = False

    # Map of Gurobi status to CVXPY status.
    STATUS_MAP = {2: s.OPTIMAL,
                  3: s.INFEASIBLE,
                  5: s.UNBOUNDED,
                  4: s.SOLVER_ERROR,
                  6: s.SOLVER_ERROR,
                  7: s.SOLVER_ERROR,
                  8: s.SOLVER_ERROR,
                  9: s.SOLVER_ERROR,
                  10: s.SOLVER_ERROR,
                  11: s.SOLVER_ERROR,
                  12: s.SOLVER_ERROR,
                  13: s.SOLVER_ERROR}

    def name(self):
        """The name of the solver.
        """
        return s.GUROBI_LIN

    def import_solver(self):
        """Imports the solver.
        """
        import gurobipy

    def matrix_intf(self):
        """The interface for matrices passed to the solver.
        """
        return intf.DEFAULT_SPARSE_INTERFACE

    def vec_intf(self):
        """The interface for vectors passed to the solver.
        """
        return intf.DEFAULT_INTERFACE

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
        # return (constr_map[s.EQ], constr_map[s.LEQ], constr_map[s.EXP])
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
        import gurobipy

        model = gurobipy.Model()

        # Set verbosity and other parameters
        if verbose:
            model.setParam("OutputFlag", True)
        else:
            model.setParam("OutputFlag", False)

        for key, value in solver_opts.items():
            model.setParam(key, value)

        # Get problem data
        data = self.get_problem_data(objective, constraints, cached_data)

        c = data[s.C]
        b = data[s.B]
        h = data[s.H]
        A = dok_matrix(data[s.A])
        G = dok_matrix(data[s.G])

        n = c.shape[0]

        variables = [model.addVar(obj = c[i], name = "x_%d" % i) for i in xrange(n)]
        model.update()

        eq_constrs = []
        if A.nnz > 0:
            I, J = zip(*[x for x in A.iterkeys()])
            I_unique = list(set(I))
            A_nonzero_locs = gurobipy.tuplelist([x for x in A.iterkeys()])

            for i in I_unique:
                expr_list = []
                for loc in A_nonzero_locs.select(i, "*"):
                    expr_list.append((A[loc], variables[loc[1]]))
                expr = gurobipy.LinExpr(expr_list)
                eq_constrs.append(model.addConstr(expr, gurobipy.GRB.EQUAL, b[i]))

        ineq_constrs = []
        if G.nnz > 0:
            I, J = zip(*[x for x in G.iterkeys()])
            I_unique = list(set(I))
            G_nonzero_locs = gurobipy.tuplelist([x for x in G.iterkeys()])

            for i in I_unique:
                expr_list = []
                for loc in G_nonzero_locs.select(i, "*"):
                    expr_list.append((G[loc], variables[loc[1]]))
                expr = gurobipy.LinExpr(expr_list)

                ineq_constrs.append(model.addConstr(expr, gurobipy.GRB.LESS_EQUAL, h[i]))

        model.update()

        try:
            model.optimize()
            results_dict = {
                "status": self.STATUS_MAP.get(model.Status, "unknown"),
                "primal objective": model.ObjVal,
                "x": np.array([v.X for v in variables]),
                "y": np.array([lc.Pi for lc in eq_constrs]),
                "z": np.array([lc.Pi for lc in ineq_constrs]),
                }
        except gurobipy.GurobiError:
            results_dict = {
                "status": s.SOLVER_ERROR
            }



        return self.format_results(results_dict, data[s.DIMS],
                                   data[s.OFFSET], cached_data)


    def format_results(self, results_dict, dims, obj_offset, cached_data):
        """Converts the solver output into standard form.

        Parameters
        ----------
        results_dict : dict
            The solver output.
        dims : dict
            The cone dimensions in the canonicalized problem.
        obj_offset : float, optional
            The constant term in the objective.
        cached_data : dict
            A map of solver name to cached problem data.

        Returns
        -------
        dict
            The solver output in standard form.
        """
        new_results = {}
        new_results[s.STATUS] = results_dict['status']
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
