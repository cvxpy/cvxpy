import abc

class Reduction(object):
    """ Abstract base class for reductions. """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def accepts(self, problem):
        """Returns whether the reduction applies to the problem.
        """
        return NotImplemented

    @abc.abstractmethod
    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.
        """
        return NotImplemented

    @abc.abstractmethod
    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        return NotImplemented
