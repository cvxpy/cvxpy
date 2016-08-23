__author__ = 'Xinyue'

import numpy as np
from examples.extensions.dmcp.dmcp.dmcp import bcd
from cvxpy import *

n = 5
m1 = 5
m2 = 4
A = np.matrix([[-2.45, -0.90, 1.53, -1.26, 1.76],
     [-0.12, -0.44, -0.01, 0.69, 0.90],
     [2.07, -1.20, -1.14, 2.04, -0.76],
     [-0.59, 0.07, 2.91, -4.63, -1.15],
     [-0.74, -0.23, -1.19, -0.06, -2.52]])

B = np.matrix([[0.81, -0.79, 0.00, 0.00, -0.95],
     [-0.34, -0.50, 0.06, 0.22, 0.92],
     [-1.32, 1.55, -1.22, -0.77, -1.14],
     [-2.11, 0.32, 0.00, -0.83, 0.59],
     [0.31, -0.19, -1.09, 0.00, 0.00]])

C = np.matrix([[0.00, 0.00, 0.16, 0.00, -1.78],
     [1.23, -0.38, 0.75, -0.38, 0.00],
     [0.46, 0.00, -0.05, 0.00, 0.00],
     [0.00, -0.12, 0.23, -0.12, 1.14]])
theta = 1e-2

P = Variable(n,n)
K = Variable(m1,m2)
alpha = Variable(1)

P.value = np.eye(n)
K.value = np.zeros((m1,m2))
alpha.value = -1

cost = norm(K,1)
constr = [np.eye(n) << P, (A+B*K*C).T*P+P*(A+B*K*C) << P*alpha*2, alpha<=-theta]
prob = Problem(Minimize(cost), constr)
prob.solve(method = 'bcd', max_iter = 200)
print "======= solution ======="
print "objective =", cost.value
print K.value
print alpha.value
