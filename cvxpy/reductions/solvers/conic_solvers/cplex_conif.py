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

from collections import namedtuple
from operator import attrgetter

import numpy as np
from scipy.sparse import dok_matrix

import cvxpy.settings as s
from cvxpy.constraints import SOC
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    ConicSolver, dims_to_solver_dict,)

# Values used to distinguish between linear and quadratic constraints.
_LIN, _QUAD = 0, 1
# For internal bookkeeping, we have to separate linear indices from
# quadratic indices. The "cpx_constrs" member of the solution will
# contain namedtuples of (constr_type, index) where constr_type is either
# _LIN or _QUAD.
_CpxConstr = namedtuple("_CpxConstr", ["constr_type", "index"])


def set_parameters(model, solver_opts) -> None:
    """Sets CPLEX parameters."""
    kwargs = sorted(solver_opts.keys())
    if "cplex_params" in kwargs:
        for param, value in solver_opts["cplex_params"].items():
            try:
                f = attrgetter(param)
                parameter = f(model.parameters)
                parameter.set(value)
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
    if kwargs:
        raise ValueError("invalid keyword-argument '{0}'".format(kwargs[0]))


def hide_solver_output(model) -> None:
    """Set CPLEX verbosity level (either on or off)."""
    # By default the output will be sent to stdout. Setting the output
    # streams to None will prevent any output from being shown.
    model.set_results_stream(None)
    model.set_warning_stream(None)
    model.set_error_stream(None)
    model.set_log_stream(None)


def _handle_solve_status(model, solstat):
    """Map CPLEX MIP solution status codes to non-MIP status codes."""
    status = model.solution.status

    # With CPLEX 12.10, some benders status codes were removed.
    if model.get_versionnumber() < 12100000:
        unbounded_status_codes = (
            status.MIP_unbounded,
            status.MIP_benders_master_unbounded,
            status.benders_master_unbounded
        )
    else:
        unbounded_status_codes = (
            status.MIP_unbounded,
        )

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
    elif solstat in unbounded_status_codes:
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


def get_status(model):
    """Map CPLEX status to CPXPY status."""
    pfeas = model.solution.is_primal_feasible()
    # NOTE: dfeas is always false for a MIP.
    dfeas = model.solution.is_dual_feasible()
    status = model.solution.status
    solstat = _handle_solve_status(model, model.solution.get_status())
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
    elif solstat == status.unbounded:
        return s.UNBOUNDED
    elif solstat == status.infeasible_or_unbounded:
        return s.INFEASIBLE_OR_UNBOUNDED
    else:
        return s.SOLVER_ERROR


class CPLEX(ConicSolver):
    """An interface for the CPLEX solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]

    def name(self):
        """The name of the solver. """
        return s.CPLEX

    def import_solver(self) -> None:
        """Imports the solver."""
        import cplex
        cplex  # For flake8

    def accepts(self, problem) -> bool:
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
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data['is_mip'] = data[s.BOOL_IDX] or data[s.INT_IDX]

        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        model = solution["model"]
        attr = {}
        if s.SOLVE_TIME in solution:
            attr[s.SOLVE_TIME] = solution[s.SOLVE_TIME]

        status = get_status(model)

        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = (model.solution.get_objective_value() +
                       inverse_data[s.OFFSET])

            x = np.array(model.solution.get_values())
            primal_vars = {inverse_data[CPLEX.VAR_ID]: x}

            if not inverse_data['is_mip']:
                # The dual values are retrieved in the order that the
                # constraints were added in solve_via_data() below. We
                # must be careful to map them to inverse_data[EQ_CONSTR]
                # followed by inverse_data[NEQ_CONSTR] accordingly.
                y = -np.array(model.solution.get_dual_values())
                dual_vars = utilities.get_dual_values(
                    y,
                    utilities.extract_dual_value,
                    inverse_data[CPLEX.EQ_CONSTR] + inverse_data[CPLEX.NEQ_CONSTR])

            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
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
            hide_solver_output(model)

        # For CVXPY, we set the qcpduals parameter here, but the user can
        # easily override it via the "cplex_params" solver option (see
        # set_parameters function).
        model.parameters.preprocessing.qcpduals.set(
            model.parameters.preprocessing.qcpduals.values.force)

        # Set parameters
        reoptimize = solver_opts.pop('reoptimize', False)
        set_parameters(model, solver_opts)

        # Solve problem
        solution = {"model": model}
        try:
            start_time = model.get_time()
            model.solve()
            solution[s.SOLVE_TIME] = model.get_time() - start_time

            ambiguous_status = get_status(model) == s.INFEASIBLE_OR_UNBOUNDED
            if ambiguous_status and reoptimize:
                model.parameters.preprocessing.presolve.set(0)
                start_time = model.get_time()
                model.solve()
                solution[s.SOLVE_TIME] += model.get_time() - start_time

        except Exception:
            pass

        return solution

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
        # For better performance, we add the constraints in a batch.
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
            else:
                lin_expr_list.append(None)
            lin_rhs.append(vec[i])

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
