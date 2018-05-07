CVXPY
=====================
[![Build Status](https://travis-ci.org/cvxgrp/cvxpy.png?branch=master)](https://travis-ci.org/cvxgrp/cvxpy)
[![Build status](https://ci.appveyor.com/api/projects/status/jo7tkvc58c3hgfd7?svg=true)](https://ci.appveyor.com/project/StevenDiamond/cvxpy)

**Join the [CVXPY mailing list](https://groups.google.com/forum/#!forum/cvxpy) and use [StackOverflow](https://stackoverflow.com/questions/tagged/cvxpy) for the best CVXPY support!**

**The CVXPY documentation is at [cvxpy.org](http://www.cvxpy.org/).**

CVXPY is a Python-embedded modeling language for convex optimization problems. It allows you to express your problem in a natural way that follows the math, rather than in the restrictive standard form required by solvers.

For example, the following code solves a least-squares problem where the variable is constrained by lower and upper bounds:

```python
from cvxpy import *
import numpy

# Problem data.
m = 30
n = 20
numpy.random.seed(1)
A = numpy.random.randn(m, n)
b = numpy.random.randn(m)

# Construct the problem.
x = Variable(n)
objective = Minimize(sum_squares(A*x - b))
constraints = [0 <= x, x <= 1]
prob = Problem(objective, constraints)

# The optimal objective is returned by prob.solve().
result = prob.solve()
# The optimal value for x is stored in x.value.
print(x.value)
# The optimal Lagrange multiplier for a constraint
# is stored in constraint.dual_value.
print(constraints[0].dual_value)
```

CVXPY was designed and implemented by Steven Diamond, with input from Stephen Boyd and Eric Chu.

A tutorial and other documentation can be found at [cvxpy.org](http://www.cvxpy.org/).

This git repository holds the latest development version of CVXPY. For installation instructions, see the [install guide](http://www.cvxpy.org/en/latest/install/index.html) at [cvxpy.org](http://www.cvxpy.org/).
