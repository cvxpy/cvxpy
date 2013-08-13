import abc
import cvxpy.utilities as u

class Constraint(u.Canonicalizable):
    """
    A constraint on an optimization problem of the form
    affine == affine or affine <= affine.
    Stored internally as affine <=/== 0.
    """
    __metaclass__ = abc.ABCMeta
    # Is the constraint DCP compliant?
    @abc.abstractmethod
    def is_dcp(self):
        return NotImplemented