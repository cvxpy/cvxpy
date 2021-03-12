import cvxpy.settings as s
import cvxpy.interface as intf
from cvxpy.reductions import Solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
import numpy as np


def constrain_gurobi_infty(v):
    '''
    Limit values of vector v between +/- infinity as
    defined in the Gurobi package
    '''
    import gurobipy as grb
    n = len(v)

    for i in range(n):
        if v[i] >= 1e20:
            v[i] = grb.GRB.INFINITY
        if v[i] <= -1e20:
            v[i] = -grb.GRB.INFINITY


class GUROBI(QpSolver):
    """QP interface for the Gurobi solver"""

    MIP_CAPABLE = True

    # Map of Gurobi status to CVXPY status.
    STATUS_MAP = {2: s.OPTIMAL,
                  3: s.INFEASIBLE,
                  5: s.UNBOUNDED,
                  4: s.INFEASIBLE_INACCURATE,
                  6: s.INFEASIBLE,
                  7: s.SOLVER_ERROR,
                  8: s.SOLVER_ERROR,
                  9: s.USER_LIMIT,  # Maximum time expired
                  # TODO could be anything.
                  10: s.SOLVER_ERROR,
                  11: s.SOLVER_ERROR,
                  12: s.SOLVER_ERROR,
                  13: s.OPTIMAL_INACCURATE}

    def name(self):
        return s.GUROBI

    def import_solver(self):
        import gurobipy
        gurobipy

    def invert(self, results, inverse_data):
        model = results["model"]
        x_grb = model.getVars()
        n = len(x_grb)
        constraints_grb = model.getConstrs()
        m = len(constraints_grb)

        # Note: Gurobi does not always fill BarIterCount
        # and IterCount so better using try/except
        try:
            bar_iter_count = model.BarIterCount
        except AttributeError:
            bar_iter_count = 0
        try:
            simplex_iter_count = model.IterCount
        except AttributeError:
            simplex_iter_count = 0
        # Take the sum in case they both appear. One of them
        # will be 0 anyway
        iter_count = bar_iter_count + simplex_iter_count

        # Start populating attribute dictionary
        attr = {s.SOLVE_TIME: model.Runtime,
                s.NUM_ITERS: iter_count,
                s.EXTRA_STATS: model}

        # Map GUROBI statuses back to CVXPY statuses
        status = self.STATUS_MAP.get(model.Status, s.SOLVER_ERROR)

        if (status in s.SOLUTION_PRESENT) or (model.solCount > 0):
            opt_val = model.objVal + inverse_data[s.OFFSET]
            x = np.array([x_grb[i].X for i in range(n)])

            primal_vars = {
                GUROBI.VAR_ID:
                intf.DEFAULT_INTF.const_to_matrix(np.array(x))
            }

            # Only add duals if not a MIP.
            dual_vars = None
            if not inverse_data[GUROBI.IS_MIP]:
                y = -np.array([constraints_grb[i].Pi for i in range(m)])
                dual_vars = {GUROBI.DUAL_VAR_ID: y}

        else:
            primal_vars = None
            dual_vars = None
            opt_val = np.inf
            if status == s.UNBOUNDED:
                opt_val = -np.inf

        return Solution(status, opt_val, primal_vars, dual_vars, attr)

    def _build_model(self, vars_offsets, offsets_to_vars, id2var, verbose):
        import gurobipy as grb
        model = grb.Model()
        model.setParam("OutputFlag", verbose)
        model.setParam("QCPDual", True)

        curr_offset = 0
        num_groups = len(vars_offsets)
        for gid in range(num_groups):
            var_id = offsets_to_vars[curr_offset]
            variable = id2var[var_id]
            sz = int(variable.size)
            if variable.is_boolean():
                mvar = model.addMVar(sz, ub=1.0, lb=0.0, vtype=grb.GRB.BINARY)
            elif variable.is_integer():
                mvar = model.addMVar(sz, ub=grb.GRB.INFINITY, lb=-grb.GRB.INFINITY, vtype=grb.GRB.INTEGER)
            else:
                mvar = model.addMVar(sz, ub=grb.GRB.INFINITY, lb=-grb.GRB.INFINITY, vtype=grb.GRB.CONTINUOUS)
            if variable.value is not None:
                st = variable.value
                mvar.setAttr("Start", st.flatten('F'))
            curr_offset += sz
        return model

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        import gurobipy as grb
        # N.B. Here we assume that the matrices in data are in csc format
        P = data[s.P]
        q = data[s.Q]
        A = data[s.A].tocsr()       # Convert A matrix to csr format
        b = data[s.B]
        F = data[s.F].tocsr()       # Convert F matrix to csr format
        g = data[s.G]
        n = data['n_var']
        vars_offsets = data['var_id_to_offset']
        offsets_to_vars = {v: k for k,v in vars_offsets.items()}
        id2var = data['id_to_var']

        constrain_gurobi_infty(b)
        constrain_gurobi_infty(g)

        if 'model' not in solver_cache:
            model = self._build_model(vars_offsets, offsets_to_vars, id2var, verbose)
            solver_cache['model'] = model
            if A.shape[0] > 0:
                mconstrs_eq = model.addMConstr(A, None, grb.GRB.EQUAL, b)
                solver_cache['mconstrs_eq'] = mconstrs_eq
            if F.shape[0] > 0:
                mconstrs_le = model.addMConstr(F, None, grb.GRB.LESS_EQUAL, g)
                solver_cache['mconstrs_le'] = mconstrs_le
        else:
            model = solver_cache['model']
            if A.shape[0] > 0:
                mconstrs_eq = solver_cache['mconstrs_eq']
                mconstrs_eq.setAttr(grb.GRB.Attr.RHS, b)
            if F.shape[0] > 0:
                mconstrs_le = solver_cache['mconstrs_le']
                mconstrs_le.setAttr(grb.GRB.Attr.RHS, g)
        mobj = model.setMObjective(0.5 * P, q, 0.0)

        for key, value in solver_opts.items():
            model.setParam(key, value)

        model.update()

        # Solve problem
        results_dict = {}
        try:
            # Solve
            model.optimize()
        except Exception:  # Error in the solution
            results_dict["status"] = s.SOLVER_ERROR

        results_dict["model"] = model

        return results_dict