from cvxpy.expressions.expression import Expression, Constant
from numpy import *
import numpy
import sys

thismodule = sys.modules[__name__]

# Wrap numpy ndarray creation functions to return ndarray.
__ARRAY_CREATION = ['arange','empty','empty_like','eye','identity',
                    'ones','ones_like','zeros','zeros_like',
                    'array']
def __wrap_array_creator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs).view(ndarray)
    
    return wrapper

for func in __ARRAY_CREATION:
    setattr( thismodule, func, __wrap_array_creator(getattr(numpy, func)) )


class BinaryOpsWrapper(type):
    """
    Metaclass to wrap all binary ops for the classes created.
    http://stackoverflow.com/questions/100003/what-is-a-metaclass-in-python
    """
    _WRAPPED_OPS = ["__add__", "__sub__", "__mul__", 
                    "__eq__", "__le__", "__ge__"]
    def __new__(cls, name, bases, dct):
        for func in BinaryOpsWrapper._WRAPPED_OPS:
            dct[func] = BinaryOpsWrapper._wrap_binary_op(func)

        return type.__new__(cls, name, bases, dct)

    @staticmethod
    def _wrap_binary_op(func):
        def wrapper(self, other):
            if isinstance(other, Expression):
                return getattr(Constant(self), func)(other)
            else:
                return getattr(self, func)(other)

        return wrapper

class ndarray(numpy.ndarray):
    """ 
    A wrapper on numpy.ndarray so that builtin operators work with
    Expressions as expected.
    """
    __metaclass__ = BinaryOpsWrapper

class matrix(numpy.matrix):
    """ 
    A wrapper on numpy.matrix so that builtin operators work with
    Expressions as expected.
    """
    __metaclass__ = BinaryOpsWrapper