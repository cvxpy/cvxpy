"""
Copyright 2017 Steven Diamond

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
import cvxpy.lin_ops.lin_utils as lu
import numpy as np
from cvxpy.problems.solvers.solver import Solver
from scipy.sparse import dok_matrix


class GUROBI(Solver):
    """An interface for the Gurobi solver.
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = True
    SDP_CAPABLE = False
    EXP_CAPABLE = False
    MIP_CAPABLE = True

    # Map of Gurobi status to CVXPY status.
    STATUS_MAP = {2: s.OPTIMAL,
                  3: s.INFEASIBLE,
                  5: s.UNBOUNDED,
                  4: s.SOLVER_ERROR,
                  6: s.SOLVER_ERROR,
                  7: s.SOLVER_ERROR,
                  8: s.SOLVER_ERROR,
                  # TODO could be anything.
                  # means time expired.
                  9: s.OPTIMAL_INACCURATE,
                  10: s.SOLVER_ERROR,
                  11: s.SOLVER_ERROR,
                  12: s.SOLVER_ERROR,
                  13: s.SOLVER_ERROR}

    def name(self):
        """The name of the solver.
        """
        return s.GUROBI

    def import_solver(self):
        """Imports the solver.
        """
        import gurobipy
        gurobipy  # For flake8

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
        return (constr_map[s.EQ] + constr_map[s.LEQ], [], [])

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
        A = dok_matrix(data[s.A])
        # Save the dok_matrix.
        data[s.A] = A

        n = c.shape[0]

        solver_cache = cached_data[self.name()]

        # TODO warmstart with SOC constraints.
        if warm_start and solver_cache.prev_result is not None \
           and len(data[s.DIMS][s.SOC_DIM]) == 0:
            model = solver_cache.prev_result["model"]
            variables = solver_cache.prev_result["variables"]
            gur_constrs = solver_cache.prev_result["gur_constrs"]
            c_prev = solver_cache.prev_result["c"]
            A_prev = solver_cache.prev_result["A"]
            b_prev = solver_cache.prev_result["b"]

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
            all_constrs, _, _ = self.split_constr(sym_data.constr_map)

            # If there is a parameter in the constraints,
            # A or b may have changed.
            if self._param_in_constr(all_constrs):
                A_diff = dok_matrix(A - A_prev)
                b_diff = b - b_prev

                # Figure out which rows of A and elements of b have changed
                try:
                    I, _ = zip(*[x for x in A_diff.keys()])
                except ValueError:
                    I = []
                I_unique = list(set(I) | set(np.where(b_diff)[0]))

                nonzero_locs = gurobipy.tuplelist([x for x in A.keys()])

                # Update locations which have changed
                for i in I_unique:

                    # Remove old constraint if it exists
                    if gur_constrs[i] is not None:
                        model.remove(gur_constrs[i])
                        gur_constrs[i] = None

                    # Add new constraint
                    if len(nonzero_locs.select(i, "*")) > 0:
                        expr_list = []
                        for loc in nonzero_locs.select(i, "*"):
                            expr_list.append((A[loc], variables[loc[1]]))
                        expr = gurobipy.LinExpr(expr_list)
                        if i < data[s.DIMS][s.EQ_DIM]:
                            ctype = gurobipy.GRB.EQUAL
                        elif data[s.DIMS][s.EQ_DIM] <= i \
                                < data[s.DIMS][s.EQ_DIM] + data[s.DIMS][s.LEQ_DIM]:
                            ctype = gurobipy.GRB.LESS_EQUAL
                        gur_constrs[i] = model.addConstr(expr, ctype, b[i])

                model.update()
            else:
                # Stay consistent with Gurobi's representation of the problem
                A = A_prev
                b = b_prev

        else:
            model = gurobipy.Model()
            variables = []
            for i in range(n):
                # Set variable type.
                if i in data[s.BOOL_IDX]:
                    vtype = gurobipy.GRB.BINARY
                elif i in data[s.INT_IDX]:
                    vtype = gurobipy.GRB.INTEGER
                else:
                    vtype = gurobipy.GRB.CONTINUOUS
                variables.append(
                    model.addVar(
                        obj=c[i],
                        name="x_%d" % i,
                        vtype=vtype,
                        # Gurobi's default LB is 0 (WHY???)
                        lb=-gurobipy.GRB.INFINITY,
                        ub=gurobipy.GRB.INFINITY)
                )
            model.update()

            nonzero_locs = gurobipy.tuplelist([x for x in A.keys()])
            eq_constrs = self.add_model_lin_constr(model, variables,
                                                   range(data[s.DIMS][s.EQ_DIM]),
                                                   gurobipy.GRB.EQUAL,
                                                   nonzero_locs, A, b)
            leq_start = data[s.DIMS][s.EQ_DIM]
            leq_end = data[s.DIMS][s.EQ_DIM] + data[s.DIMS][s.LEQ_DIM]
            ineq_constrs = self.add_model_lin_constr(model, variables,
                                                     range(leq_start, leq_end),
                                                     gurobipy.GRB.LESS_EQUAL,
                                                     nonzero_locs, A, b)
            soc_start = leq_end
            soc_constrs = []
            new_leq_constrs = []
            for constr_len in data[s.DIMS][s.SOC_DIM]:
                soc_end = soc_start + constr_len
                soc_constr, new_leq, new_vars = self.add_model_soc_constr(
                    model, variables, range(soc_start, soc_end),
                    nonzero_locs, A, b
                )
                soc_constrs.append(soc_constr)
                new_leq_constrs += new_leq
                variables += new_vars
                soc_start += constr_len

            gur_constrs = eq_constrs + ineq_constrs + \
                soc_constrs + new_leq_constrs
            model.update()

        # Set verbosity and other parameters
        model.setParam("OutputFlag", verbose)
        # TODO user option to not compute duals.
        model.setParam("QCPDual", True)

        for key, value in solver_opts.items():
            model.setParam(key, value)

        results_dict = {}
        try:
            model.optimize()
            results_dict["primal objective"] = model.ObjVal
            results_dict["x"] = np.array([v.X for v in variables])

            # Only add duals if not a MIP.
            # Not sure why we need to negate the following,
            # but need to in order to be consistent with other solvers.
            if not self.is_mip(data):
                vals = []
                for lc in gur_constrs:
                    if lc is not None:
                        if isinstance(lc, gurobipy.QConstr):
                            vals.append(lc.QCPi)
                        else:
                            vals.append(lc.Pi)
                    else:
                        vals.append(0)
                results_dict["y"] = -np.array(vals)

            results_dict["status"] = self.STATUS_MAP.get(model.Status,
                                                         s.SOLVER_ERROR)
        except:
            results_dict["status"] = s.SOLVER_ERROR

        results_dict["model"] = model
        results_dict["variables"] = variables
        results_dict["gur_constrs"] = gur_constrs
        results_dict[s.SOLVE_TIME] = model.Runtime

        return self.format_results(results_dict, data, cached_data)

    def add_model_lin_constr(self, model, variables,
                             rows, ctype,
                             nonzero_locs, mat, vec):
        """Adds EQ/LEQ constraints to the model using the data from mat and vec.

        Parameters
        ----------
        model : GUROBI model
            The problem model.
        variables : list
            The problem variables.
        rows : range
            The rows to be constrained.
        ctype : GUROBI constraint type
            The type of constraint.
        nonzero_locs : GUROBI tuplelist
            A list of all the nonzero locations.
        mat : SciPy COO matrix
            The matrix representing the constraints.
        vec : NDArray
            The constant part of the constraints.

        Returns
        -------
        list
            A list of constraints.
        """
        import gurobipy
        constr = []
        for i in rows:
            expr_list = []
            for loc in nonzero_locs.select(i, "*"):
                expr_list.append((mat[loc], variables[loc[1]]))
            # Ignore empty constraints.
            if len(expr_list) > 0:
                expr = gurobipy.LinExpr(expr_list)

                constr.append(
                    model.addConstr(expr, ctype, vec[i])
                )
            else:
                constr.append(None)
        return constr

    def add_model_soc_constr(self, model, variables,
                             rows, nonzero_locs, mat, vec):
        """Adds SOC constraint to the model using the data from mat and vec.

        Parameters
        ----------
        model : GUROBI model
            The problem model.
        variables : list
            The problem variables.
        rows : range
            The rows to be constrained.
        nonzero_locs : GUROBI tuplelist
            A list of all the nonzero locations.
        mat : SciPy COO matrix
            The matrix representing the constraints.
        vec : NDArray
            The constant part of the constraints.

        Returns
        -------
        tuple
            A tuple of (QConstr, list of Constr, and list of variables).
        """
        import gurobipy
        # Assume first expression (i.e. t) is nonzero.
        lin_expr_list = []
        soc_vars = []
        for i in rows:
            expr_list = []
            for loc in nonzero_locs.select(i, "*"):
                expr_list.append((mat[loc], variables[loc[1]]))
            # Ignore empty constraints.
            lin_expr_list.append(vec[i] - gurobipy.LinExpr(expr_list))

        # Make a variable and equality constraint for each term.
        soc_vars = [
            model.addVar(
                obj=0,
                name="soc_t_%d" % rows[0],
                vtype=gurobipy.GRB.CONTINUOUS,
                lb=0,
                ub=gurobipy.GRB.INFINITY)
        ]
        for i in rows[1:]:
            soc_vars += [
                model.addVar(
                    obj=0,
                    name="soc_x_%d" % i,
                    vtype=gurobipy.GRB.CONTINUOUS,
                    lb=-gurobipy.GRB.INFINITY,
                    ub=gurobipy.GRB.INFINITY)
            ]
        model.update()

        new_lin_constrs = []
        for i, _ in enumerate(lin_expr_list):
            new_lin_constrs += [
                model.addConstr(soc_vars[i] == lin_expr_list[i])
            ]

        t_term = soc_vars[0]*soc_vars[0]
        x_term = gurobipy.quicksum([var*var for var in soc_vars[1:]])
        return (model.addQConstr(x_term <= t_term),
                new_lin_constrs,
                soc_vars)

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
        dims = data[s.DIMS]
        if results_dict["status"] != s.SOLVER_ERROR:
            solver_cache = cached_data[self.name()]
            solver_cache.prev_result = {
                "model": results_dict["model"],
                "variables": results_dict["variables"],
                "gur_constrs": results_dict["gur_constrs"],
                "c": data[s.C],
                "A": data[s.A],
                "b": data[s.B],
            }
        new_results = {}
        new_results[s.STATUS] = results_dict['status']
        new_results[s.SOLVE_TIME] = results_dict[s.SOLVE_TIME]
        if new_results[s.STATUS] in s.SOLUTION_PRESENT:
            primal_val = results_dict['primal objective']
            new_results[s.VALUE] = primal_val + data[s.OFFSET]
            new_results[s.PRIMAL] = results_dict['x']
            if not self.is_mip(data):
                new_results[s.EQ_DUAL] = results_dict["y"][0:dims[s.EQ_DIM]]
                new_results[s.INEQ_DUAL] = results_dict["y"][dims[s.EQ_DIM]:]

        return new_results
