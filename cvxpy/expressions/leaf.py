import abc
import types
import expression
from affine import AffineObjective
from collections import deque

class Leaf(expression.Expression):
    """
    A leaf node, i.e. a Variable, Constant, or Parameter.
    """
    __metaclass__ = abc.ABCMeta
    # Every multiplication queue begins with the leaf itself.
    def as_term(self):
        return (self, deque([self]))

    # Objective associated with the leaf.
    def _objective(self):
        return AffineObjective([self.as_term()], self._shape)

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