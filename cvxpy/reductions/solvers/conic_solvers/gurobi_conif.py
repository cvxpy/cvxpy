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
from cvxpy.utilities.citations import CITATION_DICT


class GUROBI(ConicSolver):
    """
    An interface for the Gurobi solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = True
    BOUNDED_VARIABLES = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]
    MI_SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS

    # Keyword arguments for the CVXPY interface.
    INTERFACE_ARGS = ["save_file", "reoptimize"]

    # Map of Gurobi status to CVXPY status.
    STATUS_MAP = {2: s.OPTIMAL,
                  3: s.INFEASIBLE,
                  4: s.INFEASIBLE_OR_UNBOUNDED,  # Triggers reoptimize.
                  5: s.UNBOUNDED,
                  6: s.SOLVER_ERROR,
                  7: s.USER_LIMIT, # ITERATION_LIMIT
                  8: s.USER_LIMIT, # NODE_LIMIT
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
        status = solution['status']

        bar_iter_count = getattr(solution["model"], "BarIterCount", 0)

        attr = {s.EXTRA_STATS: solution['model'],
                s.SOLVE_TIME: solution[s.SOLVE_TIME],
                s.NUM_ITERS: bar_iter_count}

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
        A = sp.csr_array(data[s.A])
        dims = dims_to_solver_dict(data[s.DIMS])
        lb = data[s.LOWER_BOUNDS]
        ub = data[s.UPPER_BOUNDS]

        n = c.shape[0]
        if lb is None:
            lb = np.full(n, -gurobipy.GRB.INFINITY)
        if ub is None:
            ub = np.full(n, gurobipy.GRB.INFINITY)

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

        bool_idx = data[s.BOOL_IDX]
        int_idx = data[s.INT_IDX]
        is_mip = bool(bool_idx or int_idx)
        # Assign variable types by index. Testing ``i in bool_idx`` for every
        # column is quadratic in the number of integer variables.
        vtypes = np.full(n, gurobipy.GRB.CONTINUOUS, dtype="<U1")
        if bool_idx:
            vtypes[bool_idx] = gurobipy.GRB.BINARY
        if int_idx:
            vtypes[int_idx] = gurobipy.GRB.INTEGER
        vtypes = vtypes.tolist()
        names = ["x_%d" % i for i in range(n)]

        x = model.addMVar(n, lb=lb, ub=ub, obj=c, vtype=vtypes, name=names)
        model.update()

        # Set the start value of Gurobi vars to user provided values.
        start_vals = None
        if warm_start and solver_cache is not None \
                and self.name() in solver_cache:
            old_model = solver_cache[self.name()]
            old_status = self.STATUS_MAP.get(old_model.Status,
                                             s.SOLVER_ERROR)
            if (old_status in s.SOLUTION_PRESENT) or (old_model.solCount > 0):
                # The cached model also holds the auxiliary SOC variables, which
                # are appended after the n problem variables.
                start_vals = gurobipy.MVar.fromlist(old_model.getVars()[:n]).X
        elif warm_start:
            start_vals = data['init_value'][:n]
        if start_vals is not None:
            x.Start = np.asarray(start_vals)

        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
        eq_constrs = model.addMConstr(
            A[:leq_start, :], None, gurobipy.GRB.EQUAL, b[:leq_start]
        ).tolist()
        ineq_constrs = model.addMConstr(
            A[leq_start:leq_end, :], None, gurobipy.GRB.LESS_EQUAL,
            b[leq_start:leq_end]).tolist()

        soc_dims = dims[s.SOC_DIM]
        soc_start = leq_end
        soc_constrs = []
        new_leq_constrs = []
        if soc_dims:
            soc_constrs, new_leq_constrs = self.add_model_soc_constrs(
                model, soc_start, soc_dims, A, b
            )

        # Save file (*.mst, *.sol, ect.)
        if 'save_file' in solver_opts:
            model.write(solver_opts['save_file'])

        # Set parameters
        # TODO user option to not compute duals.
        model.setParam("QCPDual", True)
        for key, value in solver_opts.items():
            # Ignore arguments unique to the CVXPY interface.
            if key not in self.INTERFACE_ARGS:
                model.setParam(key, value)

        solution = {}
        try:
            model.optimize()
            if model.Status == 4 and solver_opts.get('reoptimize', False):
                # INF_OR_UNBD. Solve again to get a definitive answer.
                model.setParam("DualReductions", 0)
                model.optimize()
            solution["value"] = model.ObjVal
            # Only the n problem variables are inverted; the auxiliary SOC
            # variables are internal to this interface.
            solution["primal"] = np.array(x.X)

            # Only add duals if not a MIP.
            # Not sure why we need to negate the following,
            # but need to in order to be consistent with other solvers.
            vals = []
            if not is_mip:
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

        # Save model for warm start.
        if solver_cache is not None:
            solver_cache[self.name()] = model

        return solution

    def add_model_soc_constrs(self, model, soc_start, soc_dims, mat, vec):
        """Adds every SOC constraint to the model in a handful of calls.

        Each cone ``|| x ||_2 <= t`` over rows ``r`` of ``mat``/``vec`` is
        modelled with one auxiliary variable per row, tied to the problem
        variables by ``aux = vec[r] - mat[r, :] @ x``, plus one quadratic
        constraint per cone. Requires the Gurobi v9.5+ matrix API.

        Parameters
        ----------
        model : GUROBI model
            The problem model.
        soc_start : int
            The first row of mat/vec belonging to an SOC constraint.
        soc_dims : list
            The length of each SOC constraint.
        mat : SciPy CSR matrix
            The matrix representing the constraints.
        vec : NDArray
            The constant part of the constraints.

        Returns
        -------
        tuple
            A tuple of (list of QConstr, list of Constr).
        """
        import gurobipy as gp

        num_rows = int(np.sum(soc_dims))
        soc_end = soc_start + num_rows
        # Position of each cone's ``t`` variable within the SOC block.
        heads = np.concatenate(([0], np.cumsum(soc_dims)[:-1])).astype(int)

        aux_lb = np.full(num_rows, -gp.GRB.INFINITY)
        aux_lb[heads] = 0.0
        aux_names = ["soc_x_%d" % i for i in range(soc_start, soc_end)]
        for head in heads:
            aux_names[head] = "soc_t_%d" % (soc_start + head)
        aux_vars = model.addMVar(num_rows, obj=0.0, lb=aux_lb,
                                 ub=gp.GRB.INFINITY, name=aux_names)

        # aux == vec - mat @ x, written as [mat | I] @ [x; aux] == vec so that
        # it can be passed to Gurobi as a single matrix constraint. Passing None
        # for the variables uses every variable in the model, in creation order.
        coupling = sp.hstack(
            [mat[soc_start:soc_end, :], sp.eye_array(num_rows, format="csr")],
            format="csr")
        new_lin_constrs = model.addMConstr(
            coupling, None, gp.GRB.EQUAL, vec[soc_start:soc_end]).tolist()

        # Gurobi has no bulk API for quadratic constraints, but each cone needs
        # only a single QuadExpr, x'x - t^2 <= 0, and hence a single addTerms
        # call: the coefficients are always -1 on the leading t followed by 1 on
        # the tail, so one shared buffer can be sliced for every cone.
        aux_list = aux_vars.tolist()
        coeffs = np.ones(int(np.max(soc_dims)))
        coeffs[0] = -1.0
        soc_constrs = []
        offset = 0
        for constr_len in soc_dims:
            block = aux_list[offset:offset + constr_len]
            expr = gp.QuadExpr()
            expr.addTerms(coeffs[:constr_len], block, block)
            soc_constrs.append(model.addQConstr(expr, gp.GRB.LESS_EQUAL, 0.0))
            offset += constr_len
        return soc_constrs, new_lin_constrs

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["GUROBI"]
