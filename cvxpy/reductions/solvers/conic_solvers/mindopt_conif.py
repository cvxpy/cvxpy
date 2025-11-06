import cvxpy.settings as s
from cvxpy.constraints import SOC
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    ConicSolver,
    dims_to_solver_dict,
)

from scipy.sparse import csr_matrix

import numpy as np


class MINDOPT(ConicSolver):
    """
    An interface for the MINDOPT solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]
    MI_SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS

    # Keyword arguments for the CVXPY interface.
    INTERFACE_ARGS = ["save_file", "reoptimize"]

    # Map of MindOpt status to CVXPY status.
    STATUS_MAP = {
        1: s.OPTIMAL,  # OPTIMAL
        2: s.INFEASIBLE,  # INFEASIBLE
        3: s.UNBOUNDED,  # UNBOUNDED
        4: s.INFEASIBLE_OR_UNBOUNDED,  # INF_OR_UBD
        5: s.USER_LIMIT,  # SUB_OPTIMAL
        6: s.USER_LIMIT,  # ITERATION_LIMIT
        7: s.USER_LIMIT,  # TIME_LIMIT
        8: s.USER_LIMIT,  # NODE_LIMIT
        9: s.USER_LIMIT,  # SOLUTION_LIMIT
        10: s.USER_LIMIT,  # STALLING_NODE_LIMIT
        11: s.USER_LIMIT,  # INTERRUPTED
    }

    def name(self):
        """
        The name of the solver.
        """
        return s.MINDOPT

    def import_solver(self) -> None:
        """
        Imports the solver.
        """
        import mindoptpy     # noqa F401

    def accepts(self, problem) -> bool:
        """
        Can MindOpt solve the problem?
        """
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
        """
        Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        import mindoptpy as mp
        data, inv_data = super(MINDOPT, self).apply(problem)
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data['is_mip'] = data[s.BOOL_IDX] or data[s.INT_IDX]

        # Add initial guess.
        data['init_value'] = utilities.stack_vals(problem.variables, mp.MDO.UNDEFINED)

        return data, inv_data

    def invert(self, solution, inverse_data):
        """
        Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']
        attr = {s.EXTRA_STATS: solution['model'],
                s.SOLVE_TIME: solution[s.SOLVE_TIME]}

        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value'] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[MINDOPT.VAR_ID]: solution['primal']}
            if "eq_dual" in solution and not inverse_data['is_mip']:
                eq_dual = utilities.get_dual_values(
                    solution['eq_dual'],
                    utilities.extract_dual_value,
                    inverse_data[MINDOPT.EQ_CONSTR])
                leq_dual = utilities.get_dual_values(
                    solution['ineq_dual'],
                    utilities.extract_dual_value,
                    inverse_data[MINDOPT.NEQ_CONSTR])
                eq_dual.update(leq_dual)
                dual_vars = eq_dual
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """
        Returns the result of the call to the solver.

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
        import mindoptpy as mp

        c = data[s.C]
        b = data[s.B]
        A = csr_matrix(data[s.A])
        A.indptr = A.indptr.astype(np.int32)
        A.indices = A.indices.astype(np.int32)
        dims = dims_to_solver_dict(data[s.DIMS])

        n = c.shape[0]

        # Create a new model
        if 'env' in solver_opts:
            default_env = solver_opts['env']
            del solver_opts['env']
            model = mp.Model(env=default_env)
        else:
            # Create MindOpt model using default (unspecified) environment
            model = mp.Model()

        # Pass through verbosity
        model.setParam("OutputFlag", verbose)

        variables = []
        for i in range(n):
            # Set variable type.
            if i in data[s.BOOL_IDX]:
                vtype = mp.MDO.BINARY
            elif i in data[s.INT_IDX]:
                vtype = mp.MDO.INTEGER
            else:
                vtype = mp.MDO.CONTINUOUS
            variables.append(
                model.addVar(
                    obj=c[i],
                    # name="x_%d" % i,
                    vtype=vtype,
                    lb=-mp.MDO.INFINITY,
                    ub=mp.MDO.INFINITY)
            )

        # Set the start value of MindOpt vars to user provided values.
        x = model.getVars()
        if warm_start and solver_cache is not None \
                and self.name() in solver_cache:
            old_model = solver_cache[self.name()]
            old_status = self.STATUS_MAP.get(old_model.Status,
                                             s.SOLVER_ERROR)
            if (old_status in s.SOLUTION_PRESENT) or (old_model.SolCount > 0):
                old_x = old_model.getVars()
                for idx in range(len(x)):
                    x[idx].start = old_x[idx].X
        elif warm_start:
            for i in range(len(x)):
                x[i].start = data['init_value'][i]

        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
        if hasattr(model, 'addMConstr'):
            eq_constrs = model.addMConstr(
                A[:leq_start, :], None, mp.MDO.EQUAL, b[:leq_start]
            ).tolist()
            ineq_constrs = model.addMConstr(
                A[leq_start:leq_end, :], None, mp.MDO.LESS_EQUAL,
                b[leq_start:leq_end]).tolist()
        else:
            eq_constrs = self.add_model_lin_constr(model, variables,
                                                   range(dims[s.EQ_DIM]),
                                                   mp.MDO.EQUAL,
                                                   A, b)
            ineq_constrs = self.add_model_lin_constr(model, variables,
                                                     range(leq_start, leq_end),
                                                     mp.MDO.LESS_EQUAL,
                                                     A, b)

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
        for key, value in solver_opts.items():
            # Ignore arguments unique to the CVXPY interface.
            if key not in self.INTERFACE_ARGS:
                model.setParam(key, value)

        solution = {}
        try:
            model.optimize()
            if model.Status == mp.MDO.INF_OR_UBD and solver_opts.get('reoptimize', False):
                # INF_OR_UNBD. Solve again to get a definitive answer.
                model.setParam("Presolve", 0)
                model.optimize()
            solution["value"] = model.ObjVal
            solution["primal"] = np.array([v.X for v in variables])

            # Only add duals if not a MIP.
            # Not sure why we need to negate the following,
            # but need to in order to be consistent with other solvers.
            vals = []
            if not (data[s.BOOL_IDX] or data[s.INT_IDX]):
                lin_constrs = eq_constrs + ineq_constrs + new_leq_constrs
                vals += model.getAttr('DualSoln', lin_constrs)
                linpart_constrs = list(map(lambda q: model.getConstrs()[q.index], soc_constrs))
                vals += model.getAttr('DualSoln', linpart_constrs)
                solution["y"] = -np.array(vals)
                solution[s.EQ_DUAL] = solution["y"][0:dims[s.EQ_DIM]]
                solution[s.INEQ_DUAL] = solution["y"][dims[s.EQ_DIM]:]
        except Exception:
            pass

        solution[s.SOLVE_TIME] = model.SolverTime
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
        """
        Adds EQ/LEQ constraints to the model using the data from mat and vec.

        Parameters
        ----------
        model : MINDOPT model
            The problem model.
        variables : list
            The problem variables.
        rows : range
            The rows to be constrained.
        ctype : MINDOPT constraint type
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
        import mindoptpy as mp

        constr = []
        for i in rows:
            start = mat.indptr[i]
            end = mat.indptr[i + 1]
            x = [variables[j] for j in mat.indices[start:end]]
            coeff = mat.data[start:end]
            expr = mp.LinExpr(coeff, x)
            constr.append(model.addLConstr(expr, ctype, vec[i]))
        return constr

    def add_model_soc_constr(self, model, variables,
                             rows, mat, vec):
        """
        Adds SOC constraint to the model using the data from mat and vec.

        Parameters
        ----------
        model : MINDOPT model
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
        import mindoptpy as mp

        # Make a variable and equality constraint for each term.
        soc_vars = [
            model.addVar(
                obj=0,
                name="soc_t_%d" % rows[0],
                vtype=mp.MDO.CONTINUOUS,
                lb=0,
                ub=mp.MDO.INFINITY)
        ]
        for i in rows[1:]:
            soc_vars += [
                model.addVar(
                    obj=0,
                    name="soc_x_%d" % i,
                    vtype=mp.MDO.CONTINUOUS,
                    lb=-mp.MDO.INFINITY,
                    ub=mp.MDO.INFINITY)
            ]

        new_lin_constrs = []
        for i, row in enumerate(rows):
            start = mat.indptr[row]
            end = mat.indptr[row + 1]
            x = [variables[j] for j in mat.indices[start:end]]
            coeff = -mat.data[start:end]
            expr = mp.LinExpr(coeff, x)
            expr.addConstant(vec[row])
            new_lin_constrs.append(model.addLConstr(soc_vars[i], mp.MDO.EQUAL, expr))

        t_term = soc_vars[0]*soc_vars[0]
        x_term = mp.QuadExpr()
        x_term.addTerms(np.ones(len(rows) - 1).tolist(), soc_vars[1:], soc_vars[1:])
        return (model.addConstr(x_term <= t_term),
                new_lin_constrs,
                soc_vars)
