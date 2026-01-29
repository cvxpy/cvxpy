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

from ctypes import c_double, c_int

import numpy as np

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.affine_qp_mixin import AffineQpMixin
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.citations import CITATION_DICT


class DAQP(AffineQpMixin, ConicSolver):
    """Conic interface for the DAQP solver.

    DAQP is a QP solver that handles quadratic objectives with affine constraints.
    This conic interface allows DAQP to be used through the standard conic pathway.

    .. note::

        DAQP allows for marking individual constraints as "soft", meaning that
        a small user-defined violation is allowed. We currently don't support
        that option. DAQP also supports MIP QPs, which can be compiled by
        CVXPY and could be supported here as a future addition.
    """

    MIP_CAPABLE = False
    REQUIRES_CONSTR = False

    # Map of DAQP exit flags to CVXPY status.
    STATUS_MAP = {
        1: s.OPTIMAL,
        2: s.OPTIMAL,  # soft optimal, returned if soft constraints enabled
        -2: s.SOLVER_ERROR,  # Cycling detected
        -1: s.INFEASIBLE,
        -3: s.UNBOUNDED,
        -4: s.USER_LIMIT,  # iter limit reached
        -5: s.SOLVER_ERROR,  # non-convex problem
        -6: s.INFEASIBLE,  # provided active set infeasible
    }

    # Solver options that can be passed to DAQP
    ALLOWED_SOLVER_OPTS = (
        'primal_tol',
        'dual_tol',
        'zero_tol',
        'pivot_tol',
        'progress_tol',
        'cycle_tol',
        'iter_limit',
        'fval_bound',
        'eps_prox',
        'eta_prox',
    )

    def name(self):
        return s.DAQP

    def import_solver(self) -> None:
        import daqp
        daqp

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts,
                       solver_cache=None):
        """Solve using DAQP.

        DAQP does not support warm-starting, so the warm_start and solver_cache
        parameters are ignored.
        """
        import daqp

        # Convert conic format to QP format
        cone_dims = data[self.DIMS]
        qp_data = self.conic_to_qp_format(data, cone_dims)

        # DAQP uses dense matrices
        H = np.array(qp_data[s.P].toarray(), dtype=c_double) if s.P in qp_data else None
        f = np.array(qp_data[s.Q], dtype=c_double)

        # Stack equality and inequality constraints
        A_eq = qp_data[s.A].toarray()
        F = qp_data[s.F].toarray()
        A = np.array(np.concatenate([A_eq, F]), dtype=c_double)

        n_eq = A_eq.shape[0]
        n_ineq = F.shape[0]

        # DAQP uses upper/lower bound format
        bupper = np.array(
            np.concatenate([qp_data[s.B], qp_data[s.G]]),
            dtype=c_double)
        blower = np.array(
            np.concatenate([qp_data[s.B], -np.inf * np.ones(n_ineq)]),
            dtype=c_double)

        # Sense flags: 5 = equality, 0 = inequality
        sense = np.array(
            np.concatenate([
                np.ones(n_eq, dtype=c_int) * 5,  # equality
                np.zeros(n_ineq, dtype=c_int),   # inequality
            ]),
            dtype=c_int)

        # Handle positive definiteness for proximal regularization
        if 'eps_prox' not in solver_opts:
            if H is not None:
                try:
                    np.linalg.cholesky(H)
                    is_positive_definite = True
                except np.linalg.LinAlgError:
                    is_positive_definite = False
            else:
                is_positive_definite = False
        else:
            is_positive_definite = False

        solver_opts = solver_opts.copy()
        solver_opts['eps_prox'] = solver_opts.get(
            'eps_prox', 0. if is_positive_definite else 1e-5)

        used_solver_opts = {
            k: solver_opts[k] for k in solver_opts
            if k in self.ALLOWED_SOLVER_OPTS}

        # Handle empty quadratic term
        if H is None:
            H = np.zeros((len(f), len(f)), dtype=c_double)

        (xstar, fval, exitflag, info) = daqp.solve(
            H, f, A, bupper, blower, sense, **used_solver_opts)

        return (xstar, fval, exitflag, info)

    def invert(self, solution, inverse_data):
        (xstar, fval, exitflag, info) = solution

        attr = {
            s.SOLVE_TIME: info['solve_time'],
            s.SETUP_TIME: info['setup_time'],
        }
        attr[s.EXTRA_STATS] = info

        # Map DAQP statuses back to CVXPY statuses
        status = self.STATUS_MAP.get(exitflag, s.SOLVER_ERROR)

        if status in s.SOLUTION_PRESENT:
            opt_val = fval + inverse_data[s.OFFSET]
            primal_vars = {
                inverse_data[self.VAR_ID]:
                intf.DEFAULT_INTF.const_to_matrix(np.array(xstar))
            }
            # Build dual vars dict keyed by constraint IDs
            # DAQP returns duals in info['lam'] for all constraints
            # Unlike the QP interface, we don't have variable bounds here
            y = np.array(info['lam'])
            n_eq = inverse_data[self.DIMS].zero
            eq_dual = utilities.get_dual_values(
                y[:n_eq],
                utilities.extract_dual_value,
                inverse_data[self.EQ_CONSTR])
            ineq_dual = utilities.get_dual_values(
                y[n_eq:],
                utilities.extract_dual_value,
                inverse_data[self.NEQ_CONSTR])
            dual_vars = {}
            dual_vars.update(eq_dual)
            dual_vars.update(ineq_dual)
            attr[s.NUM_ITERS] = info['iterations']
            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            sol = failure_solution(status, attr)
        return sol

    def cite(self, data):
        return CITATION_DICT["DAQP"]
