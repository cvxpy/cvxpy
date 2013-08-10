import abc

class Constraint(object):
    """
    A constraint on an optimization problem of the form
    affine == affine or affine <= affine.
    Stored internally as affine <=/== 0.
    """
    __metaclass__ = abc.ABCMeta

    # Convert the constraint to affine constraints.
    @abc.abstractmethod
    def canonicalize(self):
        return NotImplemented

    # Is the constraint DCP compliant?
    @abc.abstractmethod
    def is_dcp(self):
        return NotImplemented