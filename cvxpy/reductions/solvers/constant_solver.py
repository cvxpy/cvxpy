import numpy as np

import cvxpy.settings as s
from cvxpy.constraints.second_order import RSOC, SOC
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.solver import Solver


def _zero_dual(constr):
    """Return a zero dual in the representation used in ``Solution.dual_vars``.

    Mirrors the packed representations real solvers produce via
    ``utilities.get_dual_values`` / ``extract_dual_value`` and the reshape step
    in ``cone_matrix_stuffing.ConeMatrixStuffing.invert``: SOC duals stay flat with
    size equal to the packed cone size, RSOC duals use the ``[dt, du, dX]``
    list that ``RSOC.save_dual_value`` consumes, and all other constraints use a
    shape-matching array.
    """
    if isinstance(constr, SOC):
        return np.zeros(constr.size)
    if isinstance(constr, RSOC):
        n_cones = constr.args[1].size
        n_x = constr.args[2].size // n_cones
        return [np.zeros(n_cones), np.zeros(n_cones), np.zeros((n_cones, n_x))]
    return np.zeros(constr.shape)


class ConstantSolver(Solver):
    """TODO(akshayka): Documentation."""

    # Solver capabilities
    MIP_CAPABLE = True

    def accepts(self, problem) -> bool:
        return len(problem.variables()) == 0

    def apply(self, problem):
        return problem, []

    def invert(self, solution, inverse_data):
        return solution

    def name(self) -> str:
        return "CONSTANT_SOLVER"

    def import_solver(self) -> None:
        return

    def is_installed(self) -> bool:
        return True

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        return self.solve(data, warm_start, verbose, solver_opts)

    def solve(self, problem, warm_start: bool, verbose: bool, solver_opts):
        if all(c.value() for c in problem.constraints):
            dual_vars = {c.id: _zero_dual(c) for c in problem.constraints}
            return Solution(s.OPTIMAL, problem.objective.value, {}, dual_vars, {})
        else:
            return Solution(s.INFEASIBLE, None, {}, {}, {})

    def cite(self, data):
        return ""
