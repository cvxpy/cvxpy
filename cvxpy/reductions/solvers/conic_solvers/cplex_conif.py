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

import cvxpy.settings as s
from cvxpy.constraints import SOC
from cvxpy.reductions.solvers.conic_solvers.scs_conif import (SCS,
                                                              dims_to_solver_dict)
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from collections import namedtuple
from scipy.sparse import dok_matrix
import numpy as np

# Values used to distinguish between linear and quadratic constraints.
_LIN, _QUAD = 0, 1
# For internal bookkeeping, we have to separate linear indices from
# quadratic indices. The "cpx_constrs" member of the solution will
# contain namedtuples of (constr_type, index) where constr_type is either
# _LIN or _QUAD.
_CpxConstr = namedtuple("_CpxConstr", ["constr_type", "index"])


class CPLEX(SCS):
    """An interface for the CPLEX solver."""

    # Solver capabilities.
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]

    def name(self):
        """The name of the solver. """
        return s.CPLEX

    def import_solver(self):
        """Imports the solver."""
        import cplex
        cplex  # For flake8

    def accepts(self, problem):
        """Can CPLEX solve the problem?
        """
        # TODO check if is matrix stuffed.
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in CPLEX.SUPPORTED_CONSTRAINTS:
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
        data, inv_data = super(CPLEX, self).apply(problem)
        variables = problem.variables()[0]
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data['is_mip'] = data[s.BOOL_IDX] or data[s.INT_IDX]

        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']

        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value'] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[CPLEX.VAR_ID]: solution['primal']}
            if not inverse_data['is_mip']:
                eq_dual = utilities.get_dual_values(
                    solution['eq_dual'],
                    utilities.extract_dual_value,
                    inverse_data[CPLEX.EQ_CONSTR])
                leq_dual = utilities.get_dual_values(
                    solution['ineq_dual'],
                    utilities.extract_dual_value,
                    inverse_data[CPLEX.NEQ_CONSTR])
                eq_dual.update(leq_dual)
                dual_vars = eq_dual
            return Solution(status, opt_val, primal_vars, dual_vars, {})
        else:
            return failure_solution(status)

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        import cplex

        c = data[s.C]
        b = data[s.B]
        A = dok_matrix(data[s.A])
        # Save the dok_matrix.
        data[s.A] = A
        dims = dims_to_solver_dict(data[s.DIMS])

        n = c.shape[0]

        model = cplex.Cplex()
        variables = []
        # cpx_constrs will contain CpxConstr namedtuples (see above).
        cpx_constrs = []
        vtype = []
        if data[s.BOOL_IDX] or data[s.INT_IDX]:
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
                                range(dims[s.EQ_DIM]),
                                'E', A, b)]

        # Add inequality (<=) constraints
        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
        cpx_constrs += [_CpxConstr(_LIN, x)
                        for x in self.add_model_lin_constr(
                                model, variables,
                                range(leq_start, leq_end),
                                'L', A, b)]

        # Add SOC constraints
        soc_start = leq_end
        for constr_len in dims[s.SOC_DIM]:
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

        solution = {}
        start_time = model.get_time()
        solution[s.SOLVE_TIME] = -1
        try:
            model.solve()
            solution[s.SOLVE_TIME] = model.get_time() - start_time
            solution["value"] = model.solution.get_objective_value()
            solution["primal"] = np.array(model.solution.get_values(variables))
            solution["status"] = self._get_status(model)

            # Only add duals if not a MIP.
            if not (data[s.BOOL_IDX] or data[s.INT_IDX]):
                vals = []
                for con in cpx_constrs:
                    assert con.index is not None
                    if con.constr_type == _LIN:
                        vals.append(model.solution.get_dual_values(con.index))
                    else:
                        assert con.constr_type == _QUAD
                        # Quadratic constraints not queried directly.
                        vals.append(0.0)
                solution["y"] = -np.array(vals)
                solution[s.EQ_DUAL] = solution["y"][0:dims[s.EQ_DIM]]
                solution[s.INEQ_DUAL] = solution["y"][dims[s.EQ_DIM]:]
        except Exception:
            if solution[s.SOLVE_TIME] < 0.0:
                solution[s.SOLVE_TIME] = model.get_time() - start_time
            solution["status"] = s.SOLVER_ERROR

        return solution

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
