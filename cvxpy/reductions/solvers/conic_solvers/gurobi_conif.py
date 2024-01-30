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
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import SOC
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    ConicSolver,
    dims_to_solver_dict,
)


class GUROBI(ConicSolver):
    """
    An interface for the Gurobi solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]
    MI_SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS

    # Map of Gurobi status to CVXPY status.
    STATUS_MAP = {2: s.OPTIMAL,
                  3: s.INFEASIBLE,
                  4: s.INFEASIBLE_OR_UNBOUNDED,  # Triggers reoptimize.
                  5: s.UNBOUNDED,
                  6: s.SOLVER_ERROR,
                  7: s.USER_LIMIT, # ITERATION_LIMIT
                  8: s.USER_LIMIT, # NODE_LIMIT
                  # TODO could be anything.
                  # means time expired.
                  9: s.USER_LIMIT,  # TIME_LIMIT
                  10: s.USER_LIMIT, # SOLUTION_LIMIT
                  11: s.USER_LIMIT, # INTERRUPTED
                  12: s.SOLVER_ERROR, # NUMERIC
                  13: s.USER_LIMIT, # SUBOPTIMAL
                  14: s.USER_LIMIT, # INPROGRESS
                  15: s.USER_LIMIT, # USER_OBJ_LIMIT
                  16: s.USER_LIMIT, # WORK_LIMIT
                  17: s.USER_LIMIT} # MEM_LIMIT

    def name(self):
        """The name of the solver.
        """
        return s.GUROBI

    def import_solver(self) -> None:
        """Imports the solver.
        """
        import gurobipy  # noqa F401

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
        import gurobipy as grb
        data, inv_data = super(GUROBI, self).apply(problem)
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data['is_mip'] = data[s.BOOL_IDX] or data[s.INT_IDX]

        # Add initial guess.
        data['init_value'] = utilities.stack_vals(problem.variables, grb.GRB.UNDEFINED)

        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        print(f"Running invert within gurobi_confif.py ... ")
        status = solution['status']
        print(f"solution status is: {status}")
        attr = {s.EXTRA_STATS: solution['model'],
                s.SOLVE_TIME: solution[s.SOLVE_TIME]}

        print(f"SOLUTION_PRESENT: {SOLUTION_PRESENT} is being redefined")
        SOLUTION_PRESENT = [s.USER_LIMIT, s.OPTIMAL, s.OPTIMAL_INACCURATE]
        print(f"SOLUTION_PRESENT: {SOLUTION_PRESENT}")

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
            print(f"Success! Code {status}")
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            print(f"Failure! Code {status}")
            return failure_solution(status, attr)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
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

        # Create a new model
        if 'env' in solver_opts:
            # Specifies environment to create Gurobi model for control over licensing and parameters
            # https://www.gurobi.com/documentation/9.1/refman/environments.html
            default_env = solver_opts['env']
            del solver_opts['env']
            model = gurobipy.Model(env=default_env)
        else:
            # Create Gurobi model using default (unspecified) environment
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

        # Set the start value of Gurobi vars to user provided values.
        x = model.getVars()
        if warm_start and solver_cache is not None \
                and self.name() in solver_cache:
            old_model = solver_cache[self.name()]
            old_status = self.STATUS_MAP.get(old_model.Status,
                                             s.SOLVER_ERROR)
            if (old_status in s.SOLUTION_PRESENT) or (old_model.solCount > 0):
                old_x = old_model.getVars()
                for idx in range(len(x)):
                    x[idx].start = old_x[idx].X
        elif warm_start:
            for i in range(len(x)):
                x[i].start = data['init_value'][i]

        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
        if hasattr(model, 'addMConstr'):
            # Code path for Gurobi v10.0-
            eq_constrs = model.addMConstr(
                A[:leq_start, :], None, gurobipy.GRB.EQUAL, b[:leq_start]
            ).tolist()
            ineq_constrs = model.addMConstr(
                A[leq_start:leq_end, :], None, gurobipy.GRB.LESS_EQUAL,
                b[leq_start:leq_end]).tolist()
        elif hasattr(model, 'addMConstrs'):
            # Code path for Gurobi v9.0-v9.5
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

        # Save file (*.mst, *.sol, ect.)
        if 'save_file' in solver_opts:
            model.write(solver_opts['save_file'])

        # Set parameters
        # TODO user option to not compute duals.
        model.setParam("QCPDual", True)
        for key, value in solver_opts.items():
            model.setParam(key, value)

        solution = {}
        try:
            model.optimize()
            if model.Status == 4 and solver_opts.get('reoptimize', False):
                # INF_OR_UNBD. Solve again to get a definitive answer.
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
        print(f"Trying to parse status {model.Status}")
        solution["status"] = self.STATUS_MAP.get(model.Status,
                                                 s.SOLVER_ERROR)
        if solution["status"] == s.SOLVER_ERROR and model.SolCount:
            solution["status"] = s.OPTIMAL_INACCURATE
        if solution["status"] == s.USER_LIMIT and not model.SolCount:
            solution["status"] = s.INFEASIBLE_INACCURATE
        solution["model"] = model

        # Save model for warm start.
        if solver_cache is not None:
            solver_cache[self.name()] = model

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


'''
from pathlib import Path
import tempfile
import re
import logging
from typing import List, Dict, Any
from cvxpy.settings import GUROBI
from cvxpy.expressions.variable import Variable
import gurobipy


class GurobiOriginPrinter:
    @staticmethod
    def log(conflict_equations_parts, constraint_origins):
        parts_to_print = []
        for conflict_equations_part in conflict_equations_parts:
            constraint_name = conflict_equations_part["name"]
            constr_origin = constraint_origins[constraint_name]
            if isinstance(constr_origin, Variable):
                parts_to_print.append(
                    (f"{constraint_name} (var)", "", constr_origin.name(), "")
                )
            else:
                file_path = Path(constr_origin.filename)
                parts_to_print.append(
                    (
                        constraint_name,
                        f"{file_path.name}:{constr_origin.lineno}",
                        constr_origin.code_context[0].strip(),
                        ", ".join(conflict_equations_part["variables"]),
                    )
                )

        logging.info("Conflicting constraints and variables")
        max_parts_length = [max(len(p[i]) for p in parts_to_print) for i in range(4)]
        for part_to_print in parts_to_print:
            line = ": ".join(
                [p.rjust(l) for p, l in zip(part_to_print, max_parts_length)]
            )
            logging.info(line)


def _refine_conflicts_if_infeasible(problem, soln) -> List[Dict[str, Any]]:
    """Run Gurobis's conflict refiner if the solution is infeasible and log the results

    If the model is infeasible this function calls Gurobi's method for computing
    the Ireducible Inconsistent Sybsystem (IIS) for the infeasible model. The IIS
    model is then parsed and information about all conflicting constraints is
    returned

    Return:
        List[Dict[str, Any]]: A list of items with information about all
            conflicting constraints. Each item in the list contains a dictionary
            with the following items:
                "name": CVXPY name of the conflicting constraint
                "id": CVXPY ID of the conflicting constraint
                "expression": expression defining the conflicting constraint in the
                    .ilp file with the IIS
                "variables": list of variables present in the conflicting constraint
    """
    conflict_equation_parts = []

    print("_refine_conflicts_if_infeasible with problem.status = " + str(problem.status))
    if not problem.status.startswith("infeasible"):
        print("skipping conflict refiner because problem.status != infeasible")
        return []
    # if problem.status == "infeasible_inaccurate":
    #    print("skipping conflict refiner because problem.status = infeasible_inaccurate")
    #    return []

    c = soln["model"]
    c.write("infeasible.lp")
    constraint_origins = soln["constraint_origins"]
    constr_name_map = {**soln["constr_neq_name_map"], **soln["constr_eq_name_map"]}

    logging.info("Starting solution refiner.")
    try:
        c.computeIIS()
        c.write("infeasible.ilp")
    except gurobipy.GurobiError as e:
        logging.info("Failed to compute IIS with error: " + str(e))
        return []
    except:
        logging.info("Failed to compute IIS.")
        return []

    with tempfile.TemporaryDirectory() as tmpdirname:
        c.write(str(Path(tmpdirname) / "model.ilp"))

        with open(Path(tmpdirname) / "model.ilp", "r") as f:
            iis_output = f.read()

            logging.debug("IIS model:")
            logging.debug(iis_output)

            constr_regex = re.compile(
                r"(?P<constr>[_\w]+\[\d+\]):(?P<expr>[\s\d\.\w\[\]+\-\n]*<?=\s-?\d+.?\d*)"
            )
            var_regex = re.compile(r"(?P<name>[_\w]+)\[(?P<index>\d+)\]")

            constr_list = {
                m.group("constr"): m.group("expr")
                for m in constr_regex.finditer(iis_output)
            }

            for name, expr in constr_list.items():
                cvxpy_constr_name = constr_name_map[name]
                cvxpy_constr_id = int(cvxpy_constr_name.split("_")[1])

                variables = [
                    f"{v.group('name')}[{v.group('index')}]"
                    for v in var_regex.finditer(expr)
                ]
                conflict_equation_parts += [
                    {
                        "name": cvxpy_constr_name,
                        "id": cvxpy_constr_id,
                        "expression": expr,
                        "variables": variables,
                    }
                ]

    GurobiOriginPrinter.log(conflict_equation_parts, constraint_origins)

    return conflict_equation_parts


def solve_with_conflict_refiner(
    problem,
    warm_start=False,
    verbose=False,
    solver_opts={},
    gp=False,
    enforce_dpp=False,
):
    """
    Solve the problem using GUROBI. If the solution is infeasible use the conflict
    refiner and log the results. Here is an example:

        problem = cp.Problem(objective, constraints.get())
        cp.gurobi.solve_with_conflict_refiner(problem)

    Arguments
        ---------
        warm_start : bool, optional
            Value is passed through to the `chain.solve_via_data`.
        verbose : bool, optional
            Value is passed through to the `chain.solve_via_data`.
        solver_opts : dict, optional
            Value is passed through to the `chain.solve_via_data`.
        gp : bool, optional
            If True, then parses the problem as a disciplined geometric program
            instead of a disciplined convex program.
        enforce_dpp : bool, optional
            When True, a DPPError will be thrown when trying to parse a non-DPP
            problem (instead of just a warning). Defaults to False.
    """
    data, chain, inverse_data = problem.get_problem_data(
        solver=GUROBI, gp=gp, enforce_dpp=enforce_dpp
    )
    soln = chain.solve_via_data(problem, data, warm_start, True, solver_opts)
    problem.unpack_results(soln, chain, inverse_data)
    soln["conflict_equation_parts"] = _refine_conflicts_if_infeasible(problem, soln)
    return soln
'''