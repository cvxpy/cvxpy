import cvxpy.settings as s
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.solver import Solver


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
            return Solution(s.OPTIMAL, problem.objective.value, {}, {}, {})
        else:
            return Solution(s.INFEASIBLE, None, {}, {}, {})

    def cite(self, data):
        return ""
