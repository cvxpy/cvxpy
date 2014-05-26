CVXPY [![Build Status](https://travis-ci.org/cvxgrp/cvxpy.png?branch=master)](https://travis-ci.org/cvxgrp/cvxpy)
=====================
**Although this project is similar to and named the same as [CVXPY](https://code.google.com/p/cvxpy/), this version is a total rewrite and is incompatible with the old one.**

**The CVXPY documentation is at [cvxpy.org](http://www.cvxpy.org/).**

CVXPY is a Python-embedded modeling language for optimization problems. CVXPY allows you to express your problem in a natural way. It automatically transforms the problem into standard form, calls a solver, and unpacks the results.

For example, the following code solves a least-squares problem where the variable is constrained by lower and upper bounds:

```
from cvxpy import *
import cvxopt

# Problem data.
m = 30
n = 20
A = cvxopt.normal(m,n)
b = cvxopt.normal(m)

# Construct the problem.
x = Variable(n)
objective = Minimize(sum_entries(square(A*x - b)))
constraints = [0 <= x, x <= 1]
p = Problem(objective, constraints)

# The optimal objective is returned by p.solve().
result = p.solve()
# The optimal value for x is stored in x.value.
print x.value
# The optimal Lagrange multiplier for a constraint
# is stored in constraint.dual_value.
print constraints[0].dual_value
```

CVXPY was designed and implemented by Steven Diamond, with input from Stephen Boyd and Eric Chu.

A tutorial and other documentation can be found at [cvxpy.org](http://www.cvxpy.org/).

This git repository holds the latest development version of CVXPY. For installation instructions, see the [install guide](http://www.cvxpy.org/en/latest/install/index.html) at [cvxpy.org](http://www.cvxpy.org/).
