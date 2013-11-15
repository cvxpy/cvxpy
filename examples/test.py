from cvxpy import *

import cvxopt
import numpy


# Problem data.
m = 100
n = 30
A = cvxopt.normal(m,n)
b = cvxopt.normal(m)

import cProfile
# Construct the problem.
x = Variable(n)
u = m*[[1]]
t = Variable(m,m)

# objective = Minimize( sum(t) )
# constraints = [0 <= t, t <= 1]
# p = Problem(objective, constraints)

# The optimal objective is returned by p.solve().
cProfile.run("""
sum(t)
""")
# The optimal value for x is stored in x.value.
#print x.value
# The optimal Lagrange multiplier for a constraint
# is stored in constraint.dual_value.
#print constraints[0].dual_value