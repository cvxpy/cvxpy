from cvxpy.reductions.reduction import Reduction


class Chain(Reduction):
    """A logical grouping of multiple reductions into a single reduction.

    Attributes
    ----------
    reductions : list[Reduction]
        A list of reductions.
    """

    def __init__(self, reductions=[]):
        self.reductions = reductions

    def __str__(self):
        return str(self.reductions)

    def __repr__(self):
        return "Chain(reductions=%s)" % repr(self.reductions)

    def accepts(self, problem):
        """A problem is accepted if the sequence of reductions is valid.

        In particular, the i-th reduction must accept the output of the i-1th
        reduction, with the first reduction (self.reductions[0])
        in the sequence taking as input the supplied problem.

        Parameters
        ----------
        problem : Problem
            The problem to check.

        Returns
        -------
        bool
            True if the chain can be applied, False otherwise.
        """

        for r in self.reductions:
            if not r.accepts(problem):
                return False
            problem, _ = r.apply(problem)
        return True

    def apply(self, problem):
        """Applies the chain to a problem and returns an equivalent problem.

        Parameters
        ----------
        problem : Problem
            The problem to which the chain will be applied.

        Returns
        -------
        Problem or dict
            The problem yielded by applying the reductions in sequence,
            starting at self.reductions[0].
        list
            The inverse data yielded by each of the reductions.
        """
        inverse_data = []
        for r in self.reductions:
            problem, inv = r.apply(problem)
            inverse_data.append(inv)
        return problem, inverse_data

    def invert(self, solution, inverse_data):
        """Returns a solution to the original problem given the inverse_data.
        """
        for r, inv in reversed(list(zip(self.reductions, inverse_data))):
            solution = r.invert(solution, inv)
        return solution
