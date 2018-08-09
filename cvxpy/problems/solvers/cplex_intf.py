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
from collections import namedtuple

import cvxpy.interface as intf
import cvxpy.settings as s
import cvxpy.lin_ops.lin_utils as lu
import numpy as np
from cvxpy.problems.solvers.solver import Solver
from scipy.sparse import dok_matrix

# Values used to distinguish between linear and quadratic constraints.
_LIN, _QUAD = 0, 1
# For internal bookkeeping, we have to separate linear indices from
# quadratic indices. The "cpx_constrs" member of the results_dict will
# contain namedtuples of (constr_type, index) where constr_type is either
# _LIN or _QUAD.
_CpxConstr = namedtuple("_CpxConstr", ["constr_type", "index"])


class CPLEX(Solver):
    """An interface for the CPLEX solver.
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = True
    SDP_CAPABLE = False
    EXP_CAPABLE = False
    MIP_CAPABLE = True

    def name(self):
        """The name of the solver.
        """
        return s.CPLEX

    def import_solver(self):
        """Imports the solver.
        """
        import cplex
        cplex  # For flake8

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
            'cplex_params' - a dictionary where the key-value pairs are
                             composed of parameter names and parameter
                             values.
            'cplex_filename' - A string specifying the filename to which
                               the problem will be written.

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        import cplex

        # Get problem data
        data = self.get_problem_data(objective, constraints, cached_data)

        c = data[s.C]
        b = data[s.B]
        A = dok_matrix(data[s.A])
        # Save the dok_matrix.
        data[s.A] = A
        data[s.BOOL_IDX] = solver_opts[s.BOOL_IDX]
        data[s.INT_IDX] = solver_opts[s.INT_IDX]

        n = c.shape[0]

        solver_cache = cached_data[self.name()]

        # TODO: warmstart with SOC constraints.
        if warm_start and solver_cache.prev_result is not None \
           and len(data[s.DIMS][s.SOC_DIM]) == 0:
            model = solver_cache.prev_result["model"]
            variables = solver_cache.prev_result["variables"]
            # cpx_constrs contains CpxConstr namedtuples (see above).
            cpx_constrs = solver_cache.prev_result["cpx_constrs"]
            c_prev = solver_cache.prev_result["c"]
            A_prev = solver_cache.prev_result["A"]
            b_prev = solver_cache.prev_result["b"]

            # If there is a parameter in the objective, it may have changed.
            if len(lu.get_expr_params(objective)) > 0:
                c_diff = c - c_prev

                I_unique = list(set(np.where(c_diff)[0]))

                for i in I_unique:
                    model.objective.set_linear(variables[i], c[i])
            else:
                # Stay consistent with CPLEX's representation of the problem
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
                    idxs, _ = zip(*[x for x in A_diff.keys()])
                except ValueError:
                    idxs = []
                I_unique = list(set(idxs) | set(np.where(b_diff)[0]))

                # Update locations which have changed
                csr = A.tocsr()
                for i in I_unique:
                    # To update a constraint, we first disable the old
                    # constraint and then add a new constraint with the
                    # modifications. This way we don't have to worry
                    # about indices needing to shift. The old constraint
                    # is disabled by setting all coefficients and the rhs
                    # to zero.
                    #
                    # NOTE: This can change the relative order of the
                    # constraints, which can result in performance
                    # variability!

                    # Disable the old constraint if it exists.
                    assert cpx_constrs[i].index is not None
                    assert cpx_constrs[i].constr_type == _LIN
                    idx = cpx_constrs[i].index
                    tmp = model.linear_constraints.get_rows(idx)
                    model.linear_constraints.set_linear_components(
                        idx,
                        cplex.SparsePair(ind=tmp.ind, val=[0.0]*len(tmp.ind)))
                    model.linear_constraints.set_rhs(idx, 0.0)

                    # Add new constraint

                    ind = [variables[x] for x in csr[i].indices]
                    val = [x for x in csr[i].data]
                    if i < data[s.DIMS][s.EQ_DIM]:
                        ctype = "E"
                    else:
                        assert data[s.DIMS][s.EQ_DIM] <= i \
                            < data[s.DIMS][s.EQ_DIM] + data[s.DIMS][s.LEQ_DIM]
                        ctype = "L"
                    new_idx = list(model.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=ind, val=val)],
                        senses=ctype,
                        rhs=[b[i]]))[0]
                    cpx_constrs[i] = _CpxConstr(_LIN, new_idx)

            else:
                # Stay consistent with CPLEX's representation of the problem
                A = A_prev
                b = b_prev

        else:
            model = cplex.Cplex()
            variables = []
            # cpx_constrs will contain CpxConstr namedtuples (see above).
            cpx_constrs = []
            vtype = []
            if self.is_mip(data):
                for i in range(n):
                    # Set variable type.
                    if i in data[s.BOOL_IDX]:
                        vtype.append('B')
                    elif i in data[s.INT_IDX]:
                        vtype.append('I')
                    else:
                        vtype.append('C')
            else:
                # If we specify types (even with 'C'), then the problem will
                # be interpreted as a MIP. Leaving vtype as an empty list
                # here, will ensure that the problem type remains an LP.
                pass
            # Add the variables in a batch
            variables = list(model.variables.add(
                obj=[c[i] for i in range(n)],
                lb=[-cplex.infinity]*n,  # default LB is 0
                ub=[cplex.infinity]*n,
                types="".join(vtype),
                names=["x_%d" % i for i in range(n)]))

            # Add equality constraints
            cpx_constrs += [_CpxConstr(_LIN, x)
                            for x in self.add_model_lin_constr(
                                    model, variables,
                                    range(data[s.DIMS][s.EQ_DIM]),
                                    'E', A, b)]

            # Add inequality (<=) constraints
            leq_start = data[s.DIMS][s.EQ_DIM]
            leq_end = data[s.DIMS][s.EQ_DIM] + data[s.DIMS][s.LEQ_DIM]
            cpx_constrs += [_CpxConstr(_LIN, x)
                            for x in self.add_model_lin_constr(
                                    model, variables,
                                    range(leq_start, leq_end),
                                    'L', A, b)]

            # Add SOC constraints
            soc_start = leq_end
            for constr_len in data[s.DIMS][s.SOC_DIM]:
                soc_end = soc_start + constr_len
                soc_constr, new_leq, new_vars = self.add_model_soc_constr(
                    model, variables, range(soc_start, soc_end), A, b)
                cpx_constrs.append(_CpxConstr(_QUAD, soc_constr))
                cpx_constrs += [_CpxConstr(_LIN, x) for x in new_leq]
                variables += new_vars
                soc_start += constr_len

        # Set verbosity
        if not verbose:
            model.set_results_stream(None)
            model.set_warning_stream(None)
            model.set_error_stream(None)
            model.set_log_stream(None)
        else:
            # By default the output will be sent to stdout.
            pass

        # TODO: user option to not compute duals.
        model.parameters.preprocessing.qcpduals.set(
            model.parameters.preprocessing.qcpduals.values.force)

        # TODO: Parameter support is functional, but perhaps not ideal.
        # The user must pass parameter names as used in the CPLEX Python
        # API, and raw values (i.e., no enum support).
        kwargs = sorted(solver_opts.keys())
        if "cplex_params" in kwargs:
            for param, value in solver_opts["cplex_params"].items():
                try:
                    eval("model.parameters.{0}.set({1})".format(param, value))
                except AttributeError:
                    raise ValueError(
                        "invalid CPLEX parameter, value pair ({0}, {1})".format(
                            param, value))
            kwargs.remove("cplex_params")
        if "cplex_filename" in kwargs:
            filename = solver_opts["cplex_filename"]
            if filename:
                model.write(filename)
            kwargs.remove("cplex_filename")
        if s.BOOL_IDX in kwargs:
            kwargs.remove(s.BOOL_IDX)
        if s.INT_IDX in kwargs:
            kwargs.remove(s.INT_IDX)
        if kwargs:
            raise ValueError("invalid keyword-argument '{0}'".format(kwargs[0]))

        results_dict = {}
        start_time = model.get_time()
        solve_time = -1
        try:
            model.solve()
            solve_time = model.get_time() - start_time
            results_dict["primal objective"] = model.solution.get_objective_value()
            results_dict["x"] = np.array(model.solution.get_values(variables))
            results_dict["status"] = self._get_status(model)

            # Only add duals if not a MIP.
            if not self.is_mip(data):
                vals = []
                for con in cpx_constrs:
                    assert con.index is not None
                    if con.constr_type == _LIN:
                        vals.append(model.solution.get_dual_values(con.index))
                    else:
                        assert con.constr_type == _QUAD
                        # Quadratic constraints not queried directly.
                        vals.append(0.0)
                results_dict["y"] = -np.array(vals)
        except Exception:
            if solve_time < 0.0:
                solve_time = model.get_time() - start_time
            results_dict["status"] = s.SOLVER_ERROR

        results_dict["model"] = model
        results_dict["variables"] = variables
        results_dict["cpx_constrs"] = cpx_constrs
        results_dict[s.SOLVE_TIME] = solve_time

        return self.format_results(results_dict, data, cached_data)

    def _handle_solve_status(self, model, solstat):
        """Map CPLEX MIP solution status codes to non-MIP status codes."""
        status = model.solution.status
        if solstat == status.MIP_optimal:
            return status.optimal
        elif solstat == status.MIP_infeasible:
            return status.infeasible
        elif solstat in (status.MIP_time_limit_feasible,
                         status.MIP_time_limit_infeasible):
            return status.abort_time_limit
        elif solstat in (status.MIP_dettime_limit_feasible,
                         status.MIP_dettime_limit_infeasible):
            return status.abort_dettime_limit
        elif solstat in (status.MIP_abort_feasible,
                         status.MIP_abort_infeasible):
            return status.abort_user
        elif solstat == status.MIP_optimal_infeasible:
            return status.optimal_infeasible
        elif solstat == status.MIP_infeasible_or_unbounded:
            return status.infeasible_or_unbounded
        elif solstat in (status.MIP_unbounded,
                         status.MIP_benders_master_unbounded,
                         status.benders_master_unbounded):
            return status.unbounded
        elif solstat in (status.feasible_relaxed_sum,
                         status.MIP_feasible_relaxed_sum,
                         status.optimal_relaxed_sum,
                         status.MIP_optimal_relaxed_sum,
                         status.feasible_relaxed_inf,
                         status.MIP_feasible_relaxed_inf,
                         status.optimal_relaxed_inf,
                         status.MIP_optimal_relaxed_inf,
                         status.feasible_relaxed_quad,
                         status.MIP_feasible_relaxed_quad,
                         status.optimal_relaxed_quad,
                         status.MIP_optimal_relaxed_quad):
            raise AssertionError(
                "feasopt status encountered: {0}".format(solstat))
        elif solstat in (status.conflict_feasible,
                         status.conflict_minimal,
                         status.conflict_abort_contradiction,
                         status.conflict_abort_time_limit,
                         status.conflict_abort_dettime_limit,
                         status.conflict_abort_iteration_limit,
                         status.conflict_abort_node_limit,
                         status.conflict_abort_obj_limit,
                         status.conflict_abort_memory_limit,
                         status.conflict_abort_user):
            raise AssertionError(
                "conflict refiner status encountered: {0}".format(solstat))
        elif solstat == status.relaxation_unbounded:
            return status.relaxation_unbounded
        elif solstat in (status.feasible,
                         status.MIP_feasible):
            return status.feasible
        elif solstat == status.benders_num_best:
            return status.num_best
        else:
            return solstat

    def _get_status(self, model):
        """Map CPLEX status to CPXPY status."""
        pfeas = model.solution.is_primal_feasible()
        # NOTE: dfeas is always false for a MIP.
        dfeas = model.solution.is_dual_feasible()
        status = model.solution.status
        solstat = self._handle_solve_status(model, model.solution.get_status())
        if solstat in (status.node_limit_infeasible,
                       status.fail_infeasible,
                       status.mem_limit_infeasible,
                       status.fail_infeasible_no_tree,
                       status.num_best):
            return s.SOLVER_ERROR
        elif solstat in (status.abort_user,
                         status.abort_iteration_limit,
                         status.abort_time_limit,
                         status.abort_dettime_limit,
                         status.abort_obj_limit,
                         status.abort_primal_obj_limit,
                         status.abort_dual_obj_limit,
                         status.abort_relaxed,
                         status.first_order):
            if pfeas:
                return s.OPTIMAL_INACCURATE
            else:
                return s.SOLVER_ERROR
        elif solstat in (status.node_limit_feasible,
                         status.solution_limit,
                         status.populate_solution_limit,
                         status.fail_feasible,
                         status.mem_limit_feasible,
                         status.fail_feasible_no_tree,
                         status.feasible):
            if dfeas:
                return s.OPTIMAL
            else:
                return s.OPTIMAL_INACCURATE
        elif solstat in (status.optimal,
                         status.optimal_tolerance,
                         status.optimal_infeasible,
                         status.optimal_populated,
                         status.optimal_populated_tolerance):
            return s.OPTIMAL
        elif solstat in (status.infeasible,
                         status.optimal_relaxed_sum,
                         status.optimal_relaxed_inf,
                         status.optimal_relaxed_quad):
            return s.INFEASIBLE
        elif solstat in (status.feasible_relaxed_quad,
                         status.feasible_relaxed_inf,
                         status.feasible_relaxed_sum):
            return s.SOLVER_ERROR
        elif solstat == status.infeasible_or_unbounded:
            return s.INFEASIBLE
        elif solstat == status.unbounded:
            return s.UNBOUNDED
        else:
            return s.SOLVER_ERROR

    def add_model_lin_constr(self, model, variables,
                             rows, ctype, mat, vec):
        """Adds EQ/LEQ constraints to the model using the data from mat and vec.

        Parameters
        ----------
        model : CPLEX model
            The problem model.
        variables : list
            The problem variables.
        rows : range
            The rows to be constrained.
        ctype : CPLEX constraint type
            The type of constraint.
        mat : SciPy COO matrix
            The matrix representing the constraints.
        vec : NDArray
            The RHS part of the constraints.

        Returns
        -------
        list
            A list of new linear constraint indices.
        """
        constr, lin_expr, rhs = [], [], []
        csr = mat.tocsr()
        for i in rows:
            ind = [variables[x] for x in csr[i].indices]
            val = [x for x in csr[i].data]
            lin_expr.append([ind, val])
            rhs.append(vec[i])
        # For better performance, we add the contraints in a batch.
        if lin_expr:
            assert len(lin_expr) == len(rhs)
            constr.extend(list(
                model.linear_constraints.add(
                    lin_expr=lin_expr,
                    senses=ctype * len(lin_expr),
                    rhs=rhs)))
        return constr

    def add_model_soc_constr(self, model, variables,
                             rows, mat, vec):
        """Adds SOC constraint to the model using the data from mat and vec.

        Parameters
        ----------
        model : CPLEX model
            The problem model.
        variables : list
            The problem variables.
        rows : range
            The rows to be constrained.
        mat : SciPy COO matrix
            The matrix representing the constraints.
        vec : NDArray
            The RHS part of the constraints.

        Returns
        -------
        tuple
            A tuple of (a new quadratic constraint index, a list of new
            supporting linear constr indices, and a list of new
            supporting variable indices).
        """
        import cplex
        # Assume first expression (i.e. t) is nonzero.
        lin_expr_list, soc_vars, lin_rhs = [], [], []
        csr = mat.tocsr()
        for i in rows:
            ind = [variables[x] for x in csr[i].indices]
            val = [x for x in csr[i].data]
            # Ignore empty constraints.
            if ind:
                lin_expr_list.append((ind, val))
                lin_rhs.append(vec[i])
            else:
                lin_expr_list.append(None)
                lin_rhs.append(0.0)

        # Make a variable and equality constraint for each term.
        soc_vars, is_first = [], True
        for i in rows:
            if is_first:
                lb = [0.0]
                names = ["soc_t_%d" % i]
                is_first = False
            else:
                lb = [-cplex.infinity]
                names = ["soc_x_%d" % i]
            soc_vars.extend(list(model.variables.add(
                obj=[0],
                lb=lb,
                ub=[cplex.infinity],
                types="",
                names=names)))

        new_lin_constrs = []
        for i, expr in enumerate(lin_expr_list):
            if expr is None:
                ind = [soc_vars[i]]
                val = [1.0]
            else:
                ind, val = expr
                ind.append(soc_vars[i])
                val.append(1.0)
            new_lin_constrs.extend(list(
                model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=ind, val=val)],
                    senses="E",
                    rhs=[lin_rhs[i]])))

        assert len(soc_vars) > 0
        qconstr = model.quadratic_constraints.add(
            lin_expr=cplex.SparsePair(ind=[], val=[]),
            quad_expr=cplex.SparseTriple(
                ind1=soc_vars,
                ind2=soc_vars,
                val=[-1.0] + [1.0] * (len(soc_vars) - 1)),
            sense="L",
            rhs=0.0,
            name="")
        return (qconstr, new_lin_constrs, soc_vars)

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
                "cpx_constrs": results_dict["cpx_constrs"],
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
