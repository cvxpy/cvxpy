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

from typing import Any, Dict, Generic, Iterator, List, Tuple, Union
import logging

from numpy import array, ndarray
from pyscipopt.scip import ExprCons, quicksum
from scipy.sparse import dok_matrix

import cvxpy.settings as s
from cvxpy.constraints import SOC
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS, dims_to_solver_dict
from cvxpy.settings import SCIP

log = logging.getLogger(__name__)

try:
    # Try to import SCIP model for typing
    from pyscipopt.scip import Model as ScipModel
except ImportError as e:
    # If it fails continue and use a generic type instead.
    log.warning("Could not import SCIP model")
    ScipModel = Generic


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
                    solution['eq_dual'],
                    utilities.extract_dual_value,
                    inverse_data[SCIP.EQ_CONSTR])
                leq_dual = utilities.get_dual_values(
                    solution['ineq_dual'],
                    utilities.extract_dual_value,
                    inverse_data[SCIP.NEQ_CONSTR])
                eq_dual.update(leq_dual)
                dual_vars = eq_dual
            return Solution(status, opt_val, primal_vars, dual_vars, {})
        else:
            return failure_solution(status)

        return Solution(status, opt_val, primal_vars, dual_vars, {})

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
        solution = self._get_solution(
            model=model,
            variables=variables,
            constraints=constraints,
            data=data,
            dims=dims,
            solver_opts=solver_opts,
        )
        return solution

    def _define_data(self, data: Dict[str, Any]) -> Tuple:
        c = data[s.C]
        b = data[s.B]
        A = dok_matrix(data[s.A])
        # Save the dok_matrix.
        data[s.A] = A
        dims = dims_to_solver_dict(data[s.DIMS])
        return A, b, c, dims

    def _create_variables(self, model: ScipModel, data: Dict[str, Any], c: ndarray):
        n = c.shape[0]
        variables = []
        for i in range(n):
            variables.append(
                model.addVar(
                    obj=c[i],
                    name="x_{}".format(i),
                    vtype=get_variable_type(i, data),
                    lb=None,
                    ub=None,
                )
            )
        return variables

    def _add_constraints(
            self,
            model: ScipModel,
            variables: List,
            A: dok_matrix,
            b: ndarray,
            dims: Dict[str, Union[int, List]],
    ) -> List:

        # Equal constraints
        equal_constraints = self.add_model_lin_constr(
            model=model,
            variables=variables,
            rows=range(dims[s.EQ_DIM]),
            ctype="==",
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
            ctype="<=",
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

    def _get_solution(
            self,
            model: ScipModel,
            variables: List,
            constraints: List,
            data: Dict[str, Any],
            dims: Dict[str, Union[int, List]],
            solver_opts: Dict,
    ) -> Dict[str, Any]:
        # Set parameters
        # TODO user option to not compute duals.
        # TODO: paramsmodel.setParam("QCPDual", True)
        # for key, value in solver_opts.items():
        #    model.setParam(key, value)

        solution = {}
        try:
            model.optimize()
            # # Reoptimize if INF_OR_UNBD, to get definitive answer.
            # if model.getStatus() == 4:
            #     model.setParam("DualReductions", 0)
            #     model.optimize()

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
                        vals.append(model.getDualsolLinear(lc))
                    else:
                        vals.append(0)
                solution["y"] = -array(vals)
                solution[s.EQ_DUAL] = solution["y"][0:dims[s.EQ_DIM]]
                solution[s.INEQ_DUAL] = solution["y"][dims[s.EQ_DIM]:]

        except Exception as e:
            print("exception whilst solving: ", e)
            pass

        solution[s.SOLVE_TIME] = model.getSolvingTime()
        solution['status'] = model.getStatus()
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
        constr = []
        expr_list = {i: [] for i in rows}
        for (i, j), c in A.items():
            v = variables[j]
            try:
                expr_list[i].append((c, v))
            except Exception as e:
                print("exception whilst building expr list: ", e)
                pass
        for i in rows:
            # Ignore empty constraints.
            if expr_list[i]:
                e = quicksum(coeff * var for coeff, var in expr_list[i])
                cons = (e == b[i]) if ctype == '==' else (e <= b[i])
                constr.append(
                    model.addCons(cons)
                )
            else:
                constr.append(None)

        return constr

    def add_model_soc_constr(
        self,
        model: ScipModel,
        variables: List,
        rows: Iterator,
        A: dok_matrix,
        b: ndarray,
    ) -> List:
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

        TODO: Look over this, seems wrong all vars are CONT?
        """
        # Assume first expression (i.e. t) is nonzero.
        expr_list = {i: [] for i in rows}
        for (i, j), c in A.items():
            v = variables[j]
            try:
                expr_list[i].append((c, v))
            except Exception as e:
                print("exception whilst building expr_list: ", e)
                pass

        lin_expr_list = [b[i] - ExprCons(expr_list[i]) for i in rows]

        # Make a variable and equality constraint for each term.
        soc_vars = [
            model.addVar(
                obj=0,
                name="soc_t_%d" % rows[0],
                vtype="CONTINUOUS",
                lb=0,
                ub=None,
            )
        ]
        for i in rows[1:]:
            soc_vars += [
                model.addVar(
                    obj=0,
                    name="soc_x_%d" % i,
                    vtype="CONTINUOUS",
                    lb=None,
                    ub=None,
                )
            ]
        model.update()

        new_lin_constrs = []
        for i, _ in enumerate(lin_expr_list):
            new_lin_constrs += [
                model.addCons(soc_vars[i] == lin_expr_list[i])
            ]

        t_term = soc_vars[0]*soc_vars[0]
        x_term = sum([var*var for var in soc_vars[1:]])

        return (
            model.addQConstr(x_term <= t_term),
            new_lin_constrs,
            soc_vars,
        )


def get_variable_type(i: int, data: Dict[str, Any]) -> str:
    if i in data[s.BOOL_IDX]:
        return "BINARY"
    elif i in data[s.INT_IDX]:
        return "INTEGER"
    return "CONTINUOUS"
