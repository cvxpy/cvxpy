"""
Badly scaled mean-variance problem with simplex constraint.

https://github.com/cvxgrp/cvxpy/issues/45

"""
from __future__ import division, print_function, absolute_import
import numpy as np
import cvxpy

n = 3
muVector = np.matrix([7.2, 9.8, 2.5])
sigma = np.diag([100.0, 50.0, 10.0])

x = cvxpy.Variable(n)

expectedReturn = muVector*x
risk = cvxpy.quad_form(x, sigma)
riskaversion=(.5)**-20
objective = cvxpy.Maximize(expectedReturn - riskaversion*risk)
constraints = [0 <= x, sum(x)==1]

p = cvxpy.Problem(objective,constraints)
result = p.solve(verbose=True)
print(result)
for w in x.value:
    print(w)

