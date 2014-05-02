#!/usr/bin/env python

from cvxpy import *
import numpy as np
np.random.seed(5)
n = 1000
x = Variable(n)
A = np.random.randn(n, n)
b = np.random.randn(n)
f = range(100)
obj = Minimize(norm(conv(f, x) - conv(f, b).value))
constraints = [x >= 0]
prob = Problem(obj, constraints)
result = prob.solve(solver=SCS, verbose=True)
print result
