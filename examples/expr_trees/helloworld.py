#!/usr/bin/env python

from cvxpy import *
import numpy as np
np.random.seed(5)
n = 1000
x = Variable(n)
A = np.random.randn(n, n)
obj = Minimize(norm(x, 2) + norm(x, 1) + norm(x, "inf"))
constraints = [2 >= square(x)]
prob = Problem(obj, constraints)
result = prob.solve(solver=SCS, expr_tree=True, verbose=True)
print result