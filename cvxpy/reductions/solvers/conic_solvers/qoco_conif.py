"""
Copyright 2025, the CVXPY Authors

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
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver

# QOCO standard form.
# minimize   (1/2)x'Px + c'x
# subject to Ax = b
#            Gx \leq_C h <==> h - Gx \in C
#
# Inputs:
# P is quadratic cost term
# c is linear cost term
# A is equality constraint matrix
# G is conic constraint matrix
# l is dimension of nonnegative orthant
# nsoc is number of second-order cones
# q is a vector of dimensions for each second-order cone

def dims_to_solver_cones(cone_dims):

    cones = {
        'z': cone_dims.zero,
        'l': cone_dims.nonneg,
        'q': cone_dims.soc,
    }
    return cones

class QOCO(ConicSolver):
    """An interface for the QOCO solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]

    STATUS_MAP = {
                    "QOCO_SOLVED": s.OPTIMAL,
                    "QOCO_SOLVED_INACCURATE": s.OPTIMAL_INACCURATE,
                    "QOCO_MAX_ITER": s.USER_LIMIT,
                    "QOCO_NUMERICAL_ERROR": s.SOLVER_ERROR
                }

    def name(self):
        """The name of the solver.
        """
        return 'QOCO'

    def import_solver(self) -> None:
        """Imports the solver.
        """
        import qoco  # noqa F401

    def supports_quad_obj(self) -> bool:
        """QOCO supports quadratic objective with second order 
        cone constraints.
        """
        return True

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """

        attr = {}
        status = self.STATUS_MAP[str(solution.status)]
        attr[s.SOLVE_TIME] = solution.solve_time_sec + solution.setup_time_sec
        attr[s.NUM_ITERS] = solution.iters

        if status in s.SOLUTION_PRESENT:
            primal_val = solution.obj
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {
                inverse_data[QOCO.VAR_ID]: solution.x
            }
            eq_dual_vars = utilities.get_dual_values(
                solution.y,
                utilities.extract_dual_value,
                inverse_data[QOCO.EQ_CONSTR]
            )
            ineq_dual_vars = utilities.get_dual_values(
                solution.z,
                utilities.extract_dual_value,
                inverse_data[QOCO.NEQ_CONSTR]
            )

            dual_vars = {}
            dual_vars.update(eq_dual_vars)
            dual_vars.update(ineq_dual_vars)

            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """

        # Format constraints
        #
        # QOCO requires constraints to be specified in the following order:
        # 1. zero cone
        # 2. non-negative orthant
        # 3. soc

        problem, data, inv_data = self._prepare_data_and_inv_data(problem)
        p = problem.cone_dims.zero
        m = p + problem.cone_dims.nonneg + sum(problem.cone_dims.soc)

        if problem.P is None:
            c, d, A, b = problem.apply_parameters()
        else:
            P, c, d, A, b = problem.apply_parameters(quad_obj=True)
            data[s.P] = P

        inv_data[s.OFFSET] = d

        data[s.C] = c
        data[s.A] = -A[0:p, :] if p > 0 else None
        data[s.B] = b[0:p] if p > 0 else None
        data[s.G] = -A[p::, :] if m > 0 else None
        data[s.H] = b[p::] if m > 0 else None

        return data, inv_data

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        verbose : Bool
            Control the verbosity.
        solver_opts : dict
            QOCO-specific solver options.

        Returns
        -------
        The result returned by a call to qoco.solve().
        """
        import qoco

        # Get p, num_nno, nsoc, and q from cones
        cones = dims_to_solver_cones(data[ConicSolver.DIMS])
        p = cones['z'] # Number of equality constraints
        num_nno = cones['l'] # Number of non-negative orthant constraints
        q = cones['q'] # Array of second-order cone dimensions
        nsoc = len(q)  # Number of second-order cones
        m = num_nno + sum(q)
        n = len(data[s.C])

        P = data[s.P] if s.P in data.keys() else None
        A = data[s.A]
        G = data[s.G]

        solver = qoco.QOCO()
        solver.setup(n, m, p, P, data[s.C], A, data[s.B], G, data[s.H], num_nno, nsoc, q,
        verbose=verbose, **solver_opts)
        results = solver.solve()

        if solver_cache is not None:
            solver_cache[self.name()] = results

        return results
