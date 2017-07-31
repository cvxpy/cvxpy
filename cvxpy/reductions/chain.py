from cvxpy.reductions.reduction import Reduction


class Chain(Reduction):
    """A logical grouping of multiple reductions into a single reduction.

    TODO(akshayka): Attributes
    """

    def __init__(self, reductions=[]):
        self.reductions = reductions

    def accepts(self, problem):
        for r in self.reductions:
            if not r.accepts(problem):
                return False
            problem, _ = r.apply(problem)
        return True

    def apply(self, problem):
        inverse_data = []
        for r in self.reductions:
            problem, inv = r.apply(problem)
            inverse_data.append(inv)
        return problem, inverse_data

    def invert(self, solution, inverse_data):
        for r, inv in reversed(zip(self.reductions, inverse_data)):
            solution = r.invert(solution, inv)
        return solution
