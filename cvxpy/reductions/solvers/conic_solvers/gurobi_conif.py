"""
Copyright 2013 Steven Diamond, 2017 Robin Verschueren

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

import numpy as np

import cvxpy.settings as s
from cvxpy.constraints import SOC
from cvxpy.reductions.solvers.conic_solvers.scs_conif import (SCS,
                                                              dims_to_solver_dict)
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
import scipy.sparse as sp


class GUROBI(SCS):
    """
    An interface for the Gurobi solver.

    * WARNING * This implementation takes an inadvisable approach by directly
    inheriting from a concrete solver (SCS). This implementation should not be
    used as a reference when writing other solver interfaces.

    TODO: simplify this file so it doesn't inherit from SCS; see gurobi_qpif.py
        for a very clean implementation of a quadratic programming interface.
    """

    # Solver capabilities.
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]

    # Map of Gurobi status to CVXPY status.
    STATUS_MAP = {2: s.OPTIMAL,
                  3: s.INFEASIBLE,
                  4: s.SOLVER_ERROR,  # Triggers reoptimize.
                  5: s.UNBOUNDED,
                  6: s.SOLVER_ERROR,
                  7: s.SOLVER_ERROR,
                  8: s.SOLVER_ERROR,
                  # TODO could be anything.
                  # means time expired.
                  9: s.USER_LIMIT,
                  10: s.SOLVER_ERROR,
                  11: s.SOLVER_ERROR,
                  12: s.SOLVER_ERROR,
                  13: s.SOLVER_ERROR}

    def name(self):
        """The name of the solver.
        """
        return s.GUROBI

    def import_solver(self) -> None:
        """Imports the solver.
        """
        import gurobipy
        gurobipy  # For flake8

    def accepts(self, problem) -> bool:
        """Can Gurobi solve the problem?
        """
        # TODO check if is matrix stuffed.
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in self.SUPPORTED_CONSTRAINTS:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data, inv_data = super(GUROBI, self).apply(problem)
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data['is_mip'] = data[s.BOOL_IDX] or data[s.INT_IDX]

        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']
        attr = {s.EXTRA_STATS: solution['model'],
                s.SOLVE_TIME: solution[s.SOLVE_TIME]}

        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value'] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[GUROBI.VAR_ID]: solution['primal']}
            if "eq_dual" in solution and not inverse_data['is_mip']:
                eq_dual = utilities.get_dual_values(
                    solution['eq_dual'],
                    utilities.extract_dual_value,
                    inverse_data[GUROBI.EQ_CONSTR])
                leq_dual = utilities.get_dual_values(
                    solution['ineq_dual'],
                    utilities.extract_dual_value,
                    inverse_data[GUROBI.NEQ_CONSTR])
                eq_dual.update(leq_dual)
                dual_vars = eq_dual
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data used by the solver.
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

        c = data[s.C]
        b = data[s.B]
        A = sp.csr_matrix(data[s.A])
        dims = dims_to_solver_dict(data[s.DIMS])

        n = c.shape[0]

        model = gurobipy.Model()
        # Pass through verbosity
        model.setParam("OutputFlag", verbose)

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

        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
        if hasattr(model, 'addMConstrs'):
            eq_constrs = model.addMConstrs(
                A[:leq_start, :], None, gurobipy.GRB.EQUAL, b[:leq_start])
            ineq_constrs = model.addMConstrs(
                A[leq_start:leq_end, :], None, gurobipy.GRB.LESS_EQUAL, b[leq_start:leq_end])
        else:
            eq_constrs = self.add_model_lin_constr(model, variables,
                                                   range(dims[s.EQ_DIM]),
                                                   gurobipy.GRB.EQUAL,
                                                   A, b)
            ineq_constrs = self.add_model_lin_constr(model, variables,
                                                     range(leq_start, leq_end),
                                                     gurobipy.GRB.LESS_EQUAL,
                                                     A, b)

        # TODO: add all SOC constrs at once! Be careful with return values
        soc_start = leq_end
        soc_constrs = []
        new_leq_constrs = []
        for constr_len in dims[s.SOC_DIM]:
            soc_end = soc_start + constr_len
            soc_constr, new_leq, new_vars = self.add_model_soc_constr(
                model, variables, range(soc_start, soc_end),
                A, b
            )
            soc_constrs.append(soc_constr)
            new_leq_constrs += new_leq
            variables += new_vars
            soc_start += constr_len

        # Set parameters
        # TODO user option to not compute duals.
        model.setParam("QCPDual", True)
        for key, value in solver_opts.items():
            model.setParam(key, value)

        solution = {}
        try:
            model.optimize()
            # Reoptimize if INF_OR_UNBD, to get definitive answer.
            if model.Status == 4:
                model.setParam("DualReductions", 0)
                model.optimize()
            solution["value"] = model.ObjVal
            solution["primal"] = np.array([v.X for v in variables])

            # Only add duals if not a MIP.
            # Not sure why we need to negate the following,
            # but need to in order to be consistent with other solvers.
            vals = []
            if not (data[s.BOOL_IDX] or data[s.INT_IDX]):
                lin_constrs = eq_constrs + ineq_constrs + new_leq_constrs
                vals += model.getAttr('Pi', lin_constrs)
                vals += model.getAttr('QCPi', soc_constrs)
                solution["y"] = -np.array(vals)
                solution[s.EQ_DUAL] = solution["y"][0:dims[s.EQ_DIM]]
                solution[s.INEQ_DUAL] = solution["y"][dims[s.EQ_DIM]:]
        except Exception:
            pass
        solution[s.SOLVE_TIME] = model.Runtime
        solution["status"] = self.STATUS_MAP.get(model.Status,
                                                 s.SOLVER_ERROR)
        if solution["status"] == s.SOLVER_ERROR and model.SolCount:
            solution["status"] = s.OPTIMAL_INACCURATE
        if solution["status"] == s.USER_LIMIT and not model.SolCount:
            solution["status"] = s.INFEASIBLE_INACCURATE
        solution["model"] = model

        return solution

    def add_model_lin_constr(self, model, variables,
                             rows, ctype,
                             mat, vec):
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
        mat : SciPy COO matrix
            The matrix representing the constraints.
        vec : NDArray
            The constant part of the constraints.

        Returns
        -------
        list
            A list of constraints.
        """
        import gurobipy as gp

        constr = []
        for i in rows:
            start = mat.indptr[i]
            end = mat.indptr[i + 1]
            x = [variables[j] for j in mat.indices[start:end]]
            coeff = mat.data[start:end]
            expr = gp.LinExpr(coeff, x)
            constr.append(model.addLConstr(expr, ctype, vec[i]))
        return constr

    def add_model_soc_constr(self, model, variables,
                             rows, mat, vec):
        """Adds SOC constraint to the model using the data from mat and vec.

        Parameters
        ----------
        model : GUROBI model
            The problem model.
        variables : list
            The problem variables.
        rows : range
            The rows to be constrained.
        mat : SciPy COO matrix
            The matrix representing the constraints.
        vec : NDArray
            The constant part of the constraints.

        Returns
        -------
        tuple
            A tuple of (QConstr, list of Constr, and list of variables).
        """
        import gurobipy as gp

        # Make a variable and equality constraint for each term.
        soc_vars = [
            model.addVar(
                obj=0,
                name="soc_t_%d" % rows[0],
                vtype=gp.GRB.CONTINUOUS,
                lb=0,
                ub=gp.GRB.INFINITY)
        ]
        for i in rows[1:]:
            soc_vars += [
                model.addVar(
                    obj=0,
                    name="soc_x_%d" % i,
                    vtype=gp.GRB.CONTINUOUS,
                    lb=-gp.GRB.INFINITY,
                    ub=gp.GRB.INFINITY)
            ]
        model.update()

        new_lin_constrs = []
        for i, row in enumerate(rows):
            start = mat.indptr[row]
            end = mat.indptr[row + 1]
            x = [variables[j] for j in mat.indices[start:end]]
            coeff = -mat.data[start:end]
            expr = gp.LinExpr(coeff, x)
            expr.addConstant(vec[row])
            new_lin_constrs.append(model.addLConstr(soc_vars[i], gp.GRB.EQUAL, expr))

        t_term = soc_vars[0]*soc_vars[0]
        x_term = gp.QuadExpr()
        x_term.addTerms(np.ones(len(rows) - 1), soc_vars[1:], soc_vars[1:])
        return (model.addQConstr(x_term <= t_term),
                new_lin_constrs,
                soc_vars)
