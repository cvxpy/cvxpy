__author__ = 'Xinyue'

from cvxpy import *
from bcd import bcd
import numpy as np
import matplotlib.pyplot as plt

n = 20

A = np.eye(4)
B = 0.1*np.eye(4)

u_low = 1
u_high = 20

c1 = 1
c2 = 20
d = 4
m = 1
x_ini = [0,0,1,0]
x_end = [-1,-1,0,1]
Q = np.eye(4)
R = 0

x = Variable(d,n)
u = Variable(m,n-1)
cost = 0
constr = [x[:,0] == x_ini, x[:,-1] == x_end]
for t in range(n):
    cost += quad_form(x[:,t], Q)
    if t < n-1:
        constr += [c1 <= u[:,t], u[:,t] <= c2]
        right_expr = A*x[:,t]
        for i in range(m):
            right_expr += B[:,:]*x[:,t]*u[i,t]
        constr.append(x[:,t+1] == right_expr)
prob = Problem(Minimize(cost), constr)
prob.solve(method = 'bcd', solver = 'SCS', linear=False, random_ini = 1, random_times = 1)
#iter, max_slack = bcd(prob, solver = 'SCS', linear=False, random_ini = 1, random_times = 1)

#x = Variable(2,n)
#u = Variable(n)
#cost = 0
#constr = [u <= u_high, u >= u_low, x[:,0] == np.zeros((2,1)), x[:,-1] == -np.ones((2,1))]
#for l in range(n):
#    cost += norm(x[:,l])
#    if l < n-1:
#        constr.append(x[:,l+1] == A*x[:,l]+B*x[:,l]*u[l])

#prob = Problem(Minimize(cost), constr)
#iter, max_slack = bcd(prob, solver = 'SCS', linear=False, lambd = 10, random_ini = 1, random_times = 1, max_iter = 100)
print "======= solution ======="
#print "number of iterations =", iter+1
print "objective =", cost.value

plt.plot(np.array(x.value[0,:]).flatten(),np.array(x.value[1,:]).flatten())
for i in range(n-1):
    plt.plot(x.value[0,i], x.value[1,i], marker = 'o', markerfacecoloralt = [u[0,i].value/(u_high+0.001)]*3)
plt.plot(x.value[0,n-1], x.value[1,n-1], marker = 'o', markerfacecoloralt = [0]*3)
plt.show()
