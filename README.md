CVXPY
=====
CVXPY is a Python modeling language for optimization problems. CVXPY lets you express your problem in a natural way. It automatically transforms the problem into a standard form, calls a solver, and unpacks the results.

For example, consider the LASSO problem:
    minimize ||Ax-b||~2~ + \lambda||x||~1~

The problem can be expressed in CVXPY like so:

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
p = Problem(Minimize(norm2(A*x - b) + lambda*norm1(x)))

# The optimal objective is returned by p.solve().
result = p.solve()
# The optimal value for x is stored in x.value.
print x.value
```

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