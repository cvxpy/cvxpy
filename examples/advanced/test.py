from cvxpy import *

import cvxopt
import numpy as np


# # Problem data.
# m = 100
# n = 30
# A = cvxopt.normal(m,n)
# b = cvxopt.normal(m)

# import cProfile
# # Construct the problem.
# x = Variable(n)
# u = m*[[1]]
# t = Variable(m,m)

# # objective = Minimize( sum(t) )
# # constraints = [0 <= t, t <= 1]
# # p = Problem(objective, constraints)

# # The optimal objective is returned by p.solve().
# cProfile.run("""
# sum(t)
# """)
# # The optimal value for x is stored in x.value.
# #print x.value
# # The optimal Lagrange multiplier for a constraint
# # is stored in constraint.dual_value.
# #print constraints[0].dual_value

class MyMeta(type):
    def __getitem__(self, key):
        print key
        return 2

    def __len__(self):
        return 1

    def __contains__(self, obj):
        print "hello"
        return 0


class Exp(object):
    def __add__(self, other):
        return 0

    def __radd__(self, other):
        return 1

    def __rmul__(self, other):
        print 1

    __array_priority__ = 100

import numpy as np
a = np.random.random((2,2))


class Bar1(object):
    __metaclass__ = MyMeta
    def __add__(self, rhs): return 0
    def __radd__(self, rhs): return 1
    def __lt__(self, rhs): return 0
    def __le__(self, rhs): return 1
    def __eq__(self, rhs): return 2
    def __ne__(self, rhs): return 3
    def __gt__(self, rhs): return 4
    def __ge__(self, rhs): return 5

    def __array_prepare__(self):
        print "hello"
        return self
    def __array_wrap__(self):
        return self

    def __array__(self):
        print "Afafaf"
        arr = np.array([self], dtype="object")
        return arr

    __array_priority__ = 100

def override(name):
    if name == "equal":
        def ufunc(x, y):
            print y
            if isinstance(y, Bar1) or \
               isinstance(y, np.ndarray) and isinstance(y[0], Bar1):
                return NotImplemented
            return getattr(np, name)(x, y)
        return ufunc
    else:
        def ufunc(x, y):
            print y
            if isinstance(y, Bar1):
                return NotImplemented
            return getattr(np, name)(x, y)
        return ufunc

np.set_numeric_ops(
    ** {
        ufunc : override(ufunc) for ufunc in (
            "less_equal", "equal", "greater_equal"
        )
    }
)

b = Bar1()
print a == b
print a <= b
print a + b
