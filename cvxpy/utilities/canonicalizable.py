import abc

class Canonicalizable(object):
    """ Interface for objects that can be canonicalized. """
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        self._aff_obj,self._aff_constr = self.canonicalize()
        super(Canonicalizable, self).__init__()

    # Returns the objective and a shallow copy of the constraints list.
    def canonical_form(self):
        return (self._aff_obj,self._aff_constr[:])

    # Returns an affine expression and affine constraints
    # representing the expression's objective and constraints
    # as a partial optimization problem.
    # Creates new variables if necessary.
    @abc.abstractmethod
    def canonicalize(self):
        return NotImplemented