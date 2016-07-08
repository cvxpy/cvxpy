__author__ = 'Xinyue'

from cvxpy import *
from bcd import bcd
import numpy as np

n = 5
r = np.ones((n,1))/float(n)
r = np.zeros((n,1))
r[1] = 0.5
r[2] = 0.5
c = np.zeros((n,1))
c[0] = 1
c = r

refer = np.array([0,10,20,30,40])
M = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        M[i,j] = np.abs(refer[i]-refer[j])

K = np.exp(-M)


mask = np.eye(n)
un_mask = np.ones((n,n))-mask

D = Variable(n)
D.value = np.zeros((n,1))
E = Variable(n)
E.value = np.zeros((n,1))
cost = sum_entries(mul_elemwise(M,diag(D)*K*diag(E)))
#constr = [mul_elemwise(un_mask,diag(D)*K*diag(E)) == 0]
constr = [D >= 0,E >= 0, sum_entries(diag(D)*K*diag(E),axis = 0) == r.T, sum_entries((diag(D)*K*diag(E)).T, axis = 1) == c ]
prob = Problem(Minimize(cost), constr)
prob.solve(method = 'bcd', solver = 'SCS')#, mu = 0.05, rho = 1.2, lambd = 1000, random_ini  = 1, proximal = 1, max_iter = 100, mu_max = 1e12)
print cost.value
print D.value
print E.value


