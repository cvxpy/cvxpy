__author__ = 'Xinyue'

from cvxpy import *
from bcd import bcd
import numpy as np
import matplotlib.pyplot as plt

n = 8
mask = np.matrix([[1,0,0,0,0],
        [0,1,0,0,0],
        [0,0,1,0,0],
        [0,0,0,1,0],
        [0,0,0,0,1]])
mask = np.eye(n)
un_mask = np.ones((n,n))-mask
off_diag = np.matrix([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
P = Variable(n,n)
P.value = np.eye(n)
A = Variable(n,n)
A.value = np.ones((n,n))
alpha = Variable(1)
alpha.value = -1
m = 1
theta = Variable(m)

cost = alpha
#constr = [P >> np.eye(n), A.T*P+P*A << P*alpha*2, diag(A) == theta, mul_elemwise(off_diag, A) == 0, norm(theta)<=10]
constr = [P >> np.eye(n), A.T*P+P*A << P*alpha*2]#, norm(mul_elemwise(mask,A),'fro')<=1, mul_elemwise(un_mask,A) == 0]#, norm(mul_elemwise(mask,A),'fro')<=100]
prob = Problem(Minimize(cost), constr)
prob.solve(method = 'bcd', solver = 'SCS', mu = 100, random_times = 1, max_iter = 1000)
print "======= solution ======="
print "objective =", cost.value
print A.value
print P.value
print norm(mul_elemwise(mask,A),'fro').value