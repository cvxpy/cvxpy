#!/usr/bin/env python

from cvxpy import *
x = Variable()
obj = Minimize(x)
constraints = [x >= 1, x == 2]
prob = Problem(obj, constraints)
result = prob.solve(solver=SCS, expr_tree=True)
print result