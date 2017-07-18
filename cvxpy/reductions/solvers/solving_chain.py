from cvxpy.reductions.chain import Chain
from cvxpy.reductions.solvers import Solver


def construct_solving_chain(problem):
    pass

class SolvingChain(Chain):
    """TODO(akshayka): Document
    """

    def __init__(self, reductions=[]):
        if not isinstance(self.reductions[-1], Solver):
            raise ValueError("Solving chains must terminate with a Solver.")
        self.problem_reductions = self.reductions[:-1]
        self.solver = self.reductions[-1]

    def solve(problem, warm_start, verbose, solver_opts):
        data, inverse_data = self.apply(problem)
        solution = self.solver.solve_via_data(data, warm_start,
                                              verbose, solver_opts)
        return self.invert(solution, inverse_data)
