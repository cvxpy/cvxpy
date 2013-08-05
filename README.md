cvxpy
=====
Supports norm2, normInf, and norm1 with vector arguments.
Supports affine expressions with variables of any dimension.

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
# Variable values are stored in the same matrix type used internally, 
# i.e. a cvxopt dense matrix.
x.value
z.value
```