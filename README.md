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

v = Variables(('x',2),('z',2))

p = Problem(
        Minimize(5 + norm1(v.z) + norm1(v.x) + normInf(v.x - v.z) ) ), 
        [v.x >= [2,3], 
         v.z <= [-1,-4], 
         norm2(v.x + v.z) <= 2]
    )

p.solve()
# Variable values are stored in the same matrix type used internally, 
# i.e. a cvxopt dense matrix.
v.x.value
v.z.value
```