cvxpy
=====
Supports norm2, normInf, and norm1 with vector arguments.
Supports affine expressions with variables of any dimension.

Currently supports cvxopt matrices, numbers, and python lists as constants. It will be easy to expand to numpy matrices, scipy, etc., as needed.

Constants are converted internally to cvxopt dense matrices. This also could be easily changed or made a user choice. The target solver is cvxopt.solvers.conelp.

Example usage (execute in python prompt from above the cvxpy directory):

from cvxpy import *

x = Variable(2)
z = Variable(2)

p = Problem(
        Minimize(5 + norm1(z) + norm1(self.x) + normInf(self.x - self.z) ) ), 
        [x >= [2,3], 
         z <= [-1,-4], 
         norm2(x + z) <= 2]
    )

p.solve()
# Variable values are stored in the same matrix type used internally, 
# i.e. a cvxopt dense matrix.
x.value
z.value