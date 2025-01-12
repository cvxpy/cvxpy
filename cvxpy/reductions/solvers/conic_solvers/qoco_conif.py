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

        # Get p, num_nn, nsoc, and q from cones
        cones = dims_to_solver_cones(data[ConicSolver.DIMS])
        p = cones['z'] # Number of equality constraints
        num_nno = cones['l'] # Number of non-negative orthant constraints
        q = cones['q'] # Array of second-order cone dimensions
        nsoc = len(q)  # Number of second-order cones
        m = num_nno + sum(q)

        # Get n, m, P, c, A, b, G, h from conic_solver's apply call
        _, n = data[s.A].shape
        if s.P in data:
            P = data[s.P]
        else:
            P = None
        c = data[s.C]

        A = data[s.A][0:p, :] if p > 0 else None
        b = data[s.B][0:p] if p > 0 else None

        G = data[s.A][p::, :] if m > 0 else None
        h = data[s.B][p::] if m > 0 else None

        solver = qoco.QOCO()
        solver.setup(n, m, p, P, c, A, b, G, h, num_nno, nsoc, q, verbose=verbose, **solver_opts)
        results = solver.solve()

        if solver_cache is not None:
            solver_cache[self.name()] = results

        return results
