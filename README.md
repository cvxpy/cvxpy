CVXPY
=====
CVXPY is a Python-embedded modeling language for optimization problems. CVXPY lets you express your problem in a natural way. It automatically transforms the problem into a standard form, calls a solver, and unpacks the results.

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
objective = Minimize(norm2(A*x - b) + lambda*norm1(x))
p = Problem(objective)

# The optimal objective is returned by p.solve().
result = p.solve()
# The optimal value for x is stored in x.value.
print x.value
```

The general form for constructing a CVXPY problem is `Problem(objective, constraints)`. The objective is either `Minimize(...)` or `Maximize(...)`. The constraints are a list of expressions of the form `... == ...`, `... <= ...`, or `... >= ...`.

For convex optimization, CVXPY problems must follow the rules of Disciplined Convex Programming (DCP). For an interactive tutorial on DCP, visit <dcp.stanford.edu>.

A and b are cvxopt matrices in the LASSO example, but that's not a requirement. CVXPY lets the user construct problem data using their library of choice. Certain libraries, such as Numpy, require a light wrapper to support operator overloading. The following code constructs A and b from Numpy ndarrays.

```
import cvxpy.numpy as np

A = np.ndarray(...)
b = np.ndarray(...)
```

You can construct a problem with changeable problem data using parameters. 

```
lambda = Parameter("positive")


Currently supports numpy arrays and matrices, cvxopt matrices, numbers, and python lists as constants. Matrices and vectors must be declared as Constants, i.e. A = Constant(numpy array). 

The alternative is that for numpy arrays and matrices, the user must use the numpy module given by cvxpy import * or import cvxpy.interface.numpy_wrapper as \<chosen name\>.

Constant values are converted internally to cvxopt dense matrices. This could be easily changed or made a user choice. The target solver is cvxopt.solvers.conelp.

Example usage (execute in python prompt from above the cvxpy directory):

```
from cvxpy import *

x = Variable(2, name='x')
z = Variable(2, name='z')

p = Problem(
        Minimize(5 + norm1(z) + norm1(x) + normInf(x - z) ) ), 
        [x >= [2,3], 
         z <= [-1,-4], 
         norm2(x + z) <= 2]
    )

p.solve()
# Variable values are stored in a cvxopt dense matrix.
x.value
z.value
```