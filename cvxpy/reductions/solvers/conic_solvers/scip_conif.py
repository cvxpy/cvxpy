"""
Copyright, the CVXPY authors

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

from typing import Any, Dict, Generic, Iterator, List, Optional, Tuple, Union
import logging

from numpy import array, ndarray
from pyscipopt import SCIP_PARAMSETTING
from pyscipopt.scip import ExprCons, quicksum
from scipy.sparse import dok_matrix

import cvxpy.settings as s
from cvxpy.constraints import SOC
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS, dims_to_solver_dict
# from cvxpy.settings import SCIP

log = logging.getLogger(__name__)

try:
    # Try to import SCIP model for typing
    from pyscipopt.scip import Model as ScipModel
except ImportError as e:
    # If it fails continue and use a generic type instead.
    log.warning("Could not import SCIP model")
    ScipModel = Generic


STATUS_MAP = {
    # # Mapping of SCIP to cvxpy status codes
    # SOLUTION_PRESENT
    "optimal": s.OPTIMAL,
    "timelimit": s.OPTIMAL_INACCURATE,
    "gaplimit": s.OPTIMAL_INACCURATE,
    "bestsollimit": s.OPTIMAL_INACCURATE,
    # INF_OR_UNB
    "infeasible": s.INFEASIBLE,
    "unbounded": s.UNBOUNDED,
    "inforunbd": s.UNBOUNDED_INACCURATE,
    # ERROR
    "userinterrupt": s.USER_LIMIT,
    "memlimit": s.USER_LIMIT,
    "sollimit": s.USER_LIMIT,
    "nodelimit": s.USER_LIMIT,
    "totalnodelimit": s.USER_LIMIT,
    "stallnodelimit": s.USER_LIMIT,
    "restartlimit": s.USER_LIMIT,
    "unknown": s.SOLVER_ERROR,
}

class ConstraintTypes:
    """Constraint type constants."""
    EQUAL = "EQUAL"
    LESS_THAN_OR_EQUAL = "LESS_THAN_OR_EQUAL"


class VariableTypes:
    """Variable type constants."""
    BINARY = "BINARY"
    INTEGER = "INTEGER"
    CONTINUOUS = "CONTINUOUS"


class SCIP(SCS):
    """An interface to the SCIP solver."""

    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]

    def name(self) -> str:
        """The name of the solver."""
        return 'SCIP'

    def import_solver(self) -> None:
        """Imports the solver."""
        from pyscipopt import scip
        scip  # For flake8

    def apply(self, problem: ParamConeProg) -> Tuple[Dict, Dict]:
        """Returns a new problem and data for inverting the new solution."""

        data, inv_data = super(SCIP, self).apply(problem)
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data['is_mip'] = data[s.BOOL_IDX] or data[s.INT_IDX]

        return data, inv_data

    def invert(self, solution: Dict[str, Any], inverse_data: Dict[str, Any]) -> Solution:
        """Returns the solution to the original problem given the inverse_data."""

        status = solution['status']
        dual_vars = None

        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value'] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[SCIP.VAR_ID]: solution['primal']}

            if "eq_dual" in solution and not inverse_data['is_mip']:
                eq_dual = utilities.get_dual_values(
                    result_vec=solution['eq_dual'],
                    parse_func=utilities.extract_dual_value,
                    constraints=inverse_data[SCIP.EQ_CONSTR],
                )
                leq_dual = utilities.get_dual_values(
                    result_vec=solution['ineq_dual'],
                    parse_func=utilities.extract_dual_value,
                    constraints=inverse_data[SCIP.NEQ_CONSTR],
                )

                eq_dual.update(leq_dual)
                dual_vars = eq_dual

            return Solution(status, opt_val, primal_vars, dual_vars, {})
        else:
            return failure_solution(status)

    def solve_via_data(
            self,
            data: Dict[str, Any],
            warm_start: bool,
            verbose: bool,
            solver_opts: Dict[str, Any],
            solver_cache: Dict = None,
    ) -> Solution:
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
        solver_cache: Dict
            # TODO: understand better
        Returns
        -------
            A solution object of the form: (status, optimal value, primal, equality,
            tuple dual, inequality dual)
        """
        from pyscipopt.scip import Model

        model = Model()
        # Pass through verbosity
        # TODO: how to pass verbosity -> model.setParam("OutputFlag", verbose)

        A, b, c, dims = self._define_data(
            data=data,
        )
        variables = self._create_variables(
            model=model,
            data=data,
            c=c
        )
        constraints = self._add_constraints(
            model=model,
            variables=variables,
            A=A,
            b=b,
            dims=dims,
        )
        self._set_params(
            model=model,
            verbose=verbose,
            solver_opts=solver_opts,
        )
        solution = self._solve(
            model=model,
            variables=variables,
            constraints=constraints,
            data=data,
            dims=dims,
            solver_opts=solver_opts,
        )
        return solution

    def _define_data(self, data: Dict[str, Any]) -> Tuple:
        """Define data parts from the data reference."""
        c = data[s.C]
        b = data[s.B]
        A = dok_matrix(data[s.A])
        # Save the dok_matrix.
        data[s.A] = A
        dims = dims_to_solver_dict(data[s.DIMS])
        return A, b, c, dims

    def _create_variables(self, model: ScipModel, data: Dict[str, Any], c: ndarray) -> List:
        """Create a list of variables."""
        return [
            model.addVar(
                obj=obj,
                name="x_%d" % n,
                vtype=get_variable_type(n=n, data=data),
                lb=None,
                ub=None,
            )
            for n, obj in enumerate(c)
        ]

    def _add_constraints(
            self,
            model: ScipModel,
            variables: List,
            A: dok_matrix,
            b: ndarray,
            dims: Dict[str, Union[int, List]],
    ) -> List:
        """Create a list of constraints."""

        # Equal constraints
        equal_constraints = self.add_model_lin_constr(
            model=model,
            variables=variables,
            rows=range(dims[s.EQ_DIM]),
            ctype=ConstraintTypes.EQUAL,
            A=A,
            b=b,
        )

        # Less than or equal constraints
        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
        inequal_constraints = self.add_model_lin_constr(
            model=model,
            variables=variables,
            rows=range(leq_start, leq_end),
            ctype=ConstraintTypes.LESS_THAN_OR_EQUAL,
            A=A,
            b=b,
        )

        # Second Order Cone constraints
        soc_start = leq_end
        soc_constrs = []
        new_leq_constrs = []
        for constr_len in dims[s.SOC_DIM]:
            soc_end = soc_start + constr_len
            soc_constr, new_leq, new_vars = self.add_model_soc_constr(
                model=model,
                variables=variables,
                rows=range(soc_start, soc_end),
                A=A,
                b=b,
            )
            soc_constrs.append(soc_constr)
            new_leq_constrs += new_leq
            variables += new_vars
            soc_start += constr_len

        return equal_constraints + inequal_constraints + new_leq_constrs + soc_constrs

    def _set_params(self, model: ScipModel, verbose: bool, solver_opts: Optional[Dict] = None) -> None:
        """Set model solve parameters."""

        # Default parameters:
        # These settings are needed  to allow the dual to be calculated
        model.setPresolve(SCIP_PARAMSETTING.OFF)
        model.setHeuristics(SCIP_PARAMSETTING.OFF)
        model.disablePropagation()

        # Set model verbosity
        hide_output = not verbose
        model.hideOutput(hide_output)

        # User-input parameters
        if solver_opts:
            model.setParams(**solver_opts)

    def _solve(
            self,
            model: ScipModel,
            variables: List,
            constraints: List,
            data: Dict[str, Any],
            dims: Dict[str, Union[int, List]],
            solver_opts: Dict,
    ) -> Dict[str, Any]:
        """Solve and return a solution if one exists."""

        solution = {}
        try:
            model.optimize()

            solution["value"] = model.getObjVal()
            sol = model.getBestSol()
            solution["primal"] = array([sol[v] for v in variables])

            # Only add duals if not a MIP.
            # Not sure why we need to negate the following,
            # but need to in order to be consistent with other solvers.
            if not (data[s.BOOL_IDX] or data[s.INT_IDX]):
                vals = []
                for lc in constraints:
                    if lc is not None:
                        dual = model.getDualsolLinear(lc)
                        vals.append(dual)
                    else:
                        vals.append(0)
                solution["y"] = -array(vals)
                solution[s.EQ_DUAL] = solution["y"][0:dims[s.EQ_DIM]]
                solution[s.INEQ_DUAL] = solution["y"][dims[s.EQ_DIM]:]

        except:
            pass

        solution[s.SOLVE_TIME] = model.getSolvingTime()
        solution['status'] = STATUS_MAP[model.getStatus()]
        if solution["status"] == s.SOLVER_ERROR and model.SolCount:
            solution["status"] = s.OPTIMAL_INACCURATE

        return solution

    def add_model_lin_constr(
        self,
        model: ScipModel,
        variables: List,
        rows: Iterator,
        ctype: str,
        A: dok_matrix,
        b: ndarray,
    ) -> List:
        """Adds EQ/LEQ constraints to the model using the data from mat and vec.

        Parameters
        ----------
        model : model
            The problem model.
        variables : list
            The problem variables.
        rows : range
            The rows to be constrained.
        ctype : constraint type
            The type of constraint.
        A : SciPy COO matrix
            The matrix representing the constraints.
        b : NDArray
            The constant part of the constraints.

        Returns
        -------
        list
            A list of constraints.
        """
        constraints = []

        expr_list = {i: [] for i in rows}
        for (i, j), c in A.items():
            v = variables[j]
            try:
                expr_list[i].append((c, v))
            except:
                pass

        for i in rows:
            # Ignore empty constraints.
            if expr_list[i]:
                expression = quicksum(coeff * var for coeff, var in expr_list[i])
                constraint = model.addCons(
                    (expression == b[i])
                    if ctype == ConstraintTypes.EQUAL
                    else (expression <= b[i])
                )
                constraints.append(constraint)
            else:
                constraints.append(None)

        return constraints

    def add_model_soc_constr(
        self,
        model: ScipModel,
        variables: List,
        rows: Iterator,
        A: dok_matrix,
        b: ndarray,
    ) -> Tuple:
        """Adds SOC constraint to the model using the data from mat and vec.

        Parameters
        ----------
        model : model
            The problem model.
        variables : list
            The problem variables.
        rows : range
            The rows to be constrained.
        A : SciPy COO matrix
            The matrix representing the constraints.
        b : NDArray
            The constant part of the constraints.

        Returns
        -------
        tuple
            A tuple of (QConstr, list of Constr, and list of variables).
        """

        # Assume first expression (i.e. t) is nonzero.
        expr_list = {i: [] for i in rows}
        for (i, j), c in A.items():
            v = variables[j]
            try:
                expr_list[i].append((c, v))
            except:
                pass

        # Make a variable and equality constraint for each term.
        soc_vars = [
            # TODO: Check not a gurobi thing? i.e just the first with lb=0
            model.addVar(
                obj=0,
                name="soc_t_%d" % rows[0],
                vtype="CONTINUOUS",
                lb=0,
                ub=None,
            )
        ] + [
            model.addVar(
                obj=0,
                name="soc_x_%d" % i,
                vtype="CONTINUOUS",
                lb=None,
                ub=None,
            )
            for i in rows[1:]
        ]

        lin_expr_list = [
            b[i] - quicksum(coeff * var for coeff, var in expr_list[i])
            for i in rows
        ]

        new_lin_constrs = [
            model.addCons(soc_vars[i] == lin_expr_list[i])
            for i, _ in enumerate(lin_expr_list)
        ]

        # Interesting because only <=?
        t_term = soc_vars[0] * soc_vars[0]
        x_term = quicksum([var * var for var in soc_vars[1:]])
        constraint = model.addCons(x_term <= t_term)

        return (
            constraint,
            new_lin_constrs,
            soc_vars,
        )


def get_variable_type(n: int, data: Dict[str, Any]) -> str:
    """Given an index n, and a set of data,
    return the type of a variable with the same index."""
    if n in data[s.BOOL_IDX]:
        return VariableTypes.BINARY
    elif n in data[s.INT_IDX]:
        return VariableTypes.INTEGER
    return VariableTypes.CONTINUOUS
