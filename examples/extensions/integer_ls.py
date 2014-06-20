from cvxpy import *
from ncvx.boolean import Boolean
import ncvx.branch_and_bound
import cvxopt

x = Boolean(3, name='x')
A = cvxopt.matrix([1,2,3,4,5,6,7,8,9], (3, 3), tc='d')
z = cvxopt.matrix([3, 7, 9])

p = Problem(Minimize(sum_squares(A*x - z))).solve(method="branch and bound")

print x.value
print p

# even a simple problem like this introduces too many variables
# y = Boolean()
# Problem(Minimize(square(y - 0.5))).branch_and_bound()
