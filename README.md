CVXPY
=====================
What is CVXPY?
---------------------
CVXPY is a Python-embedded modeling language for optimization problems. CVXPY lets you express your problem in a natural way. It automatically transforms the problem into standard form, calls a solver, and unpacks the results.

For example, here's a standard LASSO problem in CVXPY:

```
from cvxpy import *

# Problem data.
n = 10
m = 5
A = cvxopt.normal(n,m)
b = cvxopt.normal(m)
lambda = 1

# Construct the problem.
x = Variable(m)
objective = Minimize(sum(square(A*x - b)) + lambda*norm1(x))
p = Problem(objective)

# The optimal objective is returned by p.solve().
result = p.solve()
# The optimal value for x is stored in x.value.
print x.value
```

The general form for constructing a CVXPY problem is `Problem(objective, constraints)`. The objective is either `Minimize(...)` or `Maximize(...)`. The constraints are a list of expressions of the form `... == ...`, `... <= ...`, or `... >= ...`.

For convex optimization, CVXPY problems must follow the rules of Disciplined Convex Programming (DCP). For an interactive tutorial on DCP, visit <http://dcp.stanford.edu/>.

Prerequisites
---------------------
CVXPY requires:
* Python 2.7
* [CVXOPT](http://abel.ee.ucla.edu/cvxopt/)
* [ECOS](http://github.com/ifa-ethz/ecos)

To run the unit tests, you need
* [Nose](http://nose.readthedocs.org)

#Installation


Features
=====================

Problem Data
---------------------
CVXPY lets you construct problem data using your library of choice. Certain libraries, such as Numpy, require a light wrapper to support operator overloading. The following code constructs A and b from Numpy ndarrays.

```
import cvxpy.numpy as np

A = np.ndarray(...)
b = np.ndarray(...)
```

A natural extension to the LASSO example is to construct a tradeoff curve of the least squares penalty vs. the cardinality of x for different values of lambda. You can do this efficiently in CVXPY using parameters. The value of a Parameter can be initialized and changed after the problem is constructed. Here's the LASSO example with lambda as a parameter:

```
import cvxpy.numpy as np

# Problem data.
...
lambda = Parameter("positive")

# Construct the problem.
x = Variable(m)
objective = Minimize(sum(square(A*x - b)) + lambda*norm1(x))
p = Problem(objective)

# Vary lambda for trade-off curve.
x_values = []
for value in np.logspace(-1, 2, num=100):
    lambda.value = value
    p.solve()
    x_values.append(x.value)

# Construct a trade off curve using the x_values.
...
```

Parameterized problems can be solved in parallel. See examples/stock_tradeoff.py for an example.

Object Oriented Optimization
---------------------
CVXPY can be 