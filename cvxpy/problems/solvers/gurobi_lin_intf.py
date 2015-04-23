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
import cvxpy.lin_ops.lin_utils as lu
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

    CUSTOM_OPTS = [
                  "update_eq_constrs",
                  "update_ineq_constrs",
                  "update_objective",
                  ]

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
        return (constr_map[s.EQ], constr_map[s.LEQ], [])

    @staticmethod
    def _param_in_constr(constraints):
        """Do any of the constraints contain parameters?
        """
        for constr in constraints:
            if len(lu.get_expr_params(constr.expr)) > 0:
                return True
        return False

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

        # Get problem data
        data = self.get_problem_data(objective, constraints, cached_data)

        c = data[s.C]
        b = data[s.B]
        h = data[s.H]
        A = dok_matrix(data[s.A])
        G = dok_matrix(data[s.G])

        n = c.shape[0]

        solver_cache = cached_data[self.name()]

        if warm_start and solver_cache.prev_result is not None:
            model = solver_cache.prev_result["model"]
            variables = solver_cache.prev_result["variables"]
            eq_constrs = solver_cache.prev_result["eq_constrs"]
            ineq_constrs = solver_cache.prev_result["ineq_constrs"]
            c_prev = solver_cache.prev_result["c"]
            A_prev = solver_cache.prev_result["A"]
            b_prev = solver_cache.prev_result["b"]
            G_prev = solver_cache.prev_result["G"]
            h_prev = solver_cache.prev_result["h"]

            # If there is a parameter in the objective, it may have changed.
            if len(lu.get_expr_params(objective)) > 0:
                c_diff = c - c_prev

                I_unique = list(set(np.where(c_diff)[0]))

                for i in I_unique:
                    variables[i].Obj = c[i]
            else:
                # Stay consistent with Gurobi's representation of the problem
                c = c_prev

            # Get equality and inequality constraints.
            sym_data = self.get_sym_data(objective, constraints, cached_data)
            eq_constr, ineq_constr, _ = self.split_constr(sym_data.constr_map)

            # If there is a parameter in the equality constraints,
            # A or b may have changed.
            if self._param_in_constr(eq_constr):
                A_diff = dok_matrix(A - A_prev)
                b_diff = b - b_prev

                # Figure out which rows of A and elements of b have changed
                try:
                    I, _ = zip(*[x for x in A_diff.iterkeys()])
                except ValueError:
                    I = []
                I_unique = list(set(I) | set(np.where(b_diff)[0]))

                A_nonzero_locs = gurobipy.tuplelist([x for x in A.iterkeys()])

                # Update locations which have changed
                for i in I_unique:

                    # Remove old constraint if it exists
                    if eq_constrs[i] != None:
                        model.remove(eq_constrs[i])
                        eq_constrs[i] = None

                    # Add new constraint
                    if len(A_nonzero_locs.select(i, "*")):
                        expr_list = []
                        for loc in A_nonzero_locs.select(i, "*"):
                            expr_list.append((A[loc], variables[loc[1]]))
                        expr = gurobipy.LinExpr(expr_list)
                        eq_constrs[i] = model.addConstr(expr,
                                                        gurobipy.GRB.EQUAL,
                                                        b[i])

                model.update()
            else:
                # Stay consistent with Gurobi's representation of the problem
                A = A_prev
                b = b_prev

            # If there is a parameter in the inequality constraints,
            # G or h may have changed.
            if self._param_in_constr(ineq_constr):
                G_diff = dok_matrix(G - G_prev)
                h_diff = h - h_prev

                # Figure out which rows of G and elements of h have changed
                try:
                    I, _ = zip(*[x for x in G_diff.iterkeys()])
                except ValueError:
                    I = []
                I_unique = list(set(I) | set(np.where(h_diff)[0]))

                G_nonzero_locs = gurobipy.tuplelist([x for x in G.iterkeys()])

                # Update locations which have changed
                for i in I_unique:

                    # Remove old constraint if it exists
                    if ineq_constrs[i] != None:
                        model.remove(ineq_constrs[i])
                        ineq_constrs[i] = None

                    # Add new constraint
                    if len(G_nonzero_locs.select(i, "*")):
                        expr_list = []
                        for loc in G_nonzero_locs.select(i, "*"):
                            expr_list.append((G[loc], variables[loc[1]]))
                        expr = gurobipy.LinExpr(expr_list)
                        ineq_constrs[i] = model.addConstr(expr,
                                            gurobipy.GRB.LESS_EQUAL, h[i])

                model.update()
            else:
                # Stay consistent with Gurobi's representation of the problem
                G = G_prev
                h = h_prev

        else:
            model = gurobipy.Model()
            variables = [
                model.addVar(
                    obj=c[i],
                    name="x_%d" % i,
                    # Gurobi's default LB is 0 (WHY???)
                    lb=-gurobipy.GRB.INFINITY,
                    ub=gurobipy.GRB.INFINITY)
                for i in xrange(n)]
            model.update()

            eq_constrs = [None] * b.shape[0]
            if A.nnz > 0 or b.any:
                try:
                    I, _ = zip(*[x for x in A.iterkeys()])
                except ValueError:
                    I = []
                eq_constrs_nonzero_idxs = list(set(I) | set(np.where(b)[0]))
                A_nonzero_locs = gurobipy.tuplelist([x for x in A.iterkeys()])

                for i in eq_constrs_nonzero_idxs:
                    expr_list = []
                    for loc in A_nonzero_locs.select(i, "*"):
                        expr_list.append((A[loc], variables[loc[1]]))
                    expr = gurobipy.LinExpr(expr_list)

                    eq_constrs[i] = model.addConstr(expr,
                                                    gurobipy.GRB.EQUAL,
                                                    b[i])

            ineq_constrs = [None] * h.shape[0]
            if G.nnz > 0 or h.any:
                try:
                    I, _ = zip(*[x for x in G.iterkeys()])
                except ValueError:
                    I = []
                ineq_constrs_nonzero_idxs = list(set(I) | set(np.where(h)[0]))
                G_nonzero_locs = gurobipy.tuplelist([x for x in G.iterkeys()])

                for i in ineq_constrs_nonzero_idxs:
                    expr_list = []
                    for loc in G_nonzero_locs.select(i, "*"):
                        expr_list.append((G[loc], variables[loc[1]]))
                    expr = gurobipy.LinExpr(expr_list)

                    ineq_constrs[i] = model.addConstr(expr,
                                                      gurobipy.GRB.LESS_EQUAL,
                                                      h[i])

            model.update()

        # Set verbosity and other parameters
        if verbose:
            model.setParam("OutputFlag", True)
        else:
            model.setParam("OutputFlag", False)

        for key, value in solver_opts.items():
            if key in self.CUSTOM_OPTS:
                continue
            model.setParam(key, value)

        try:
            model.optimize()

            results_dict = {
                "model": model,
                "variables": variables,
                "eq_constrs": eq_constrs,
                "ineq_constrs": ineq_constrs,
                "c": c,
                "A": A,
                "b": b,
                "G": G,
                "h": h,
                "status": self.STATUS_MAP.get(model.Status, "unknown"),
                "primal objective": model.ObjVal,
                "x": np.array([v.X for v in variables]),
                # Not sure why we need to negate the following,
                # but need to in order to be consistent with other solvers.
                "y": -np.array(
                        [lc.Pi if lc != None else 0 for lc in eq_constrs]
                    ),
                "z": -np.array(
                        [lc.Pi if lc != None else 0 for lc in ineq_constrs]
                    ),
            }
        except gurobipy.GurobiError:
            results_dict = {
                "status": s.SOLVER_ERROR
            }

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
        if results_dict["status"] != s.SOLVER_ERROR:
            solver_cache = cached_data[self.name()]
            solver_cache.prev_result = {
                "model": results_dict["model"],
                "variables": results_dict["variables"],
                "eq_constrs": results_dict["eq_constrs"],
                "ineq_constrs": results_dict["ineq_constrs"],
                "c": results_dict["c"],
                "A": results_dict["A"],
                "b": results_dict["b"],
                "G": results_dict["G"],
                "h": results_dict["h"],
                }
        new_results = {}
        new_results[s.STATUS] = results_dict['status']
        if new_results[s.STATUS] in s.SOLUTION_PRESENT:
            primal_val = results_dict['primal objective']
            new_results[s.VALUE] = primal_val + data[s.OFFSET]
            new_results[s.PRIMAL] = results_dict['x']
            new_results[s.EQ_DUAL] = results_dict['y']
            new_results[s.INEQ_DUAL] = results_dict['z']

        return new_results
