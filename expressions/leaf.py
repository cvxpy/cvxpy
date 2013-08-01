import abc
from collections import deque
import types

class Leaf(object):
    """
    A leaf node, i.e. a Variable, Constant, or Parameter.
    """
    __metaclass__ = abc.ABCMeta
    # Every multiplication queue begins with the leaf itself.
    def terms(self):
        return [self]

    def canonicalize(self):
        return (self,[])

    @property
    def curvature(self):
        return self._curvature