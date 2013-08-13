import abc
import types
import expression
import cvxpy.utilities as u
from affine import AffObjective
from collections import deque

class Leaf(expression.Expression, u.Affine):
    """
    A leaf node, i.e. a Variable, Constant, or Parameter.
    """
    __metaclass__ = abc.ABCMeta
    # Objective associated with the leaf.
    def _objective(self):
        return AffObjective(self.variables(), [deque([self])], self._shape)

    # Constraints associated with the leaf.
    def _constraints(self):
        return []

    # Root for the construction of affine expressions.
    def canonicalize(self):
        return (self._objective(), self._constraints())

    # Returns the coefficients dictionary for the leaf.
    @abc.abstractmethod
    def coefficients(self, interface):
        return NotImplemented

    @property
    def size(self):
        return self._shape.size

    @property
    def sign(self):
        return self._sign