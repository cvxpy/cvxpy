"""
Copyright 2016 Sascha-Dominic Schnug

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
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    ConicSolver,
    dims_to_solver_dict,
)


class CBC(ConicSolver):
    """ An interface to the CBC solver
    """

    # Solver capabilities.
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS
    MI_SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS

    # Map of CBC status to CVXPY status.
    STATUS_MAP_MIP = {'solution': s.OPTIMAL,
                      'relaxation infeasible': s.INFEASIBLE,
                      'problem proven infeasible': s.INFEASIBLE,
                      'relaxation abandoned': s.SOLVER_ERROR,
                      'stopped on user event': s.SOLVER_ERROR,
                      'stopped on nodes': s.OPTIMAL_INACCURATE,
                      'stopped on gap': s.OPTIMAL_INACCURATE,
                      'stopped on time': s.OPTIMAL_INACCURATE,
                      'stopped on solutions': s.OPTIMAL_INACCURATE,
                      'linear relaxation unbounded': s.UNBOUNDED,
                      'unset': s.UNBOUNDED}

    STATUS_MAP_LP = {'optimal': s.OPTIMAL,
                     'primal infeasible': s.INFEASIBLE,
                     'dual infeasible': s.UNBOUNDED,
                     'stopped due to errors': s.SOLVER_ERROR,
                     'stopped by event handler (virtual int '
                     'ClpEventHandler::event())': s.SOLVER_ERROR}

    def name(self):
        """The name of the solver.
        """
        return s.CBC

    def import_solver(self) -> None:
        """Imports the solver.
        """
        from cylp.cy import CyClpSimplex  # noqa F401

    def accepts(self, problem) -> bool:
        """Can Cbc solve the problem?
        """
        # TODO check if is matrix stuffed.
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in CBC.SUPPORTED_CONSTRAINTS:
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
        data, inv_data = super(CBC, self).apply(problem)
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]

        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']

        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value'] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: solution['primal']}
            return Solution(status, opt_val, primal_vars, None, {})
        else:
            return failure_solution(status)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        # Import basic modelling tools of cylp
        from cylp.cy import CyClpSimplex
        from cylp.py.modeling.CyLPModel import CyLPArray, CyLPModel

        c = data[s.C]
        b = data[s.B]
        A = data[s.A]
        dims = dims_to_solver_dict(data[s.DIMS])

        n = c.shape[0]

        # Problem
        model = CyLPModel()

        # Variables
        x = model.addVariable('x', n)

        # Constraints
        # eq
        model += A[0:dims[s.EQ_DIM], :] * x == CyLPArray(b[0:dims[s.EQ_DIM]])

        # leq
        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
        model += A[leq_start:leq_end, :] * x <= CyLPArray(b[leq_start:leq_end])

        # Objective
        model.objective = c

        # Convert model
        model = CyClpSimplex(model)

        # No boolean vars available in Cbc -> model as int + restrict to [0,1]
        if data[s.BOOL_IDX] or data[s.INT_IDX]:
            # Mark integer- and binary-vars as "integer"
            model.setInteger(x[data[s.BOOL_IDX]])
            model.setInteger(x[data[s.INT_IDX]])

            # Restrict binary vars only
            idxs = data[s.BOOL_IDX]
            n_idxs = len(idxs)

            model.setColumnLowerSubset(np.arange(n_idxs, dtype=np.int32),
                                       np.array(idxs, np.int32),
                                       np.zeros(n_idxs))

            model.setColumnUpperSubset(np.arange(n_idxs, dtype=np.int32),
                                       np.array(idxs, np.int32),
                                       np.ones(n_idxs))

        # Verbosity Clp
        if not verbose:
            model.logLevel = 0

        # Build model & solve
        status = None
        clp_model_options = {"dualTolerance", "primalTolerance",
                             "maxNumIteration", "logLevel", "automaticScaling",
                             "scaling", "infeasibilityCost", "optimizationDirection"}
        clp_solve_options = {"presolve"}
        # all the above keys except logLevel apply only to models solved with CLP
        non_cbc_options = (clp_model_options | clp_solve_options) - {"logLevel"}
        for key in solver_opts:
            if key in clp_model_options:
                setattr(model, key, solver_opts[key])
        if data[s.BOOL_IDX] or data[s.INT_IDX]:
            # Convert model
            cbcModel = model.getCbcModel()

            # Verbosity Cbc
            if not verbose:
                cbcModel.logLevel = 0
            # Other Solver options
            for key, value in solver_opts.items():
                if key in non_cbc_options:
                    continue
                setattr(cbcModel, key, value)

            # cylp: /cylp/cy/CyCbcModel.pyx#L134
            # Call CbcMain. Solve the problem using the same parameters used by
            # CbcSolver. Equivalent to solving the model from the command line
            # using cbc's binary.
            cbcModel.solve()
            status = cbcModel.status
        else:
            # cylp: /cylp/cy/CyClpSimplex.pyx
            # Run CLP's initialSolve. It does a presolve and uses primal or dual
            # Simplex to solve a problem.
            solve_args = {}
            for key in clp_solve_options:
                if key in solver_opts:
                    solve_args[key] = solver_opts[key]
            status = model.initialSolve(**solve_args)

        solution = {}
        if data[s.BOOL_IDX] or data[s.INT_IDX]:
            solution["status"] = self.STATUS_MAP_MIP[status]
            solution["primal"] = cbcModel.primalVariableSolution['x']
            solution["value"] = cbcModel.objectiveValue
        else:
            solution["status"] = self.STATUS_MAP_LP[status]
            solution["primal"] = model.primalVariableSolution['x']
            solution["value"] = model.objectiveValue

        return solution
