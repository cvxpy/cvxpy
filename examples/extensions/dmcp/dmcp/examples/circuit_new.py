__author__ = 'Xinyue'

import numpy as np
from examples.extensions.dmcp.dmcp.dmcp import bcd
from examples.extensions.dmcp.dmcp.dmcp.find_set import find_minimal_sets
from cvxpy import *

np.random.seed(0)

n = 10
I0 = -100
delta = 1
u0 = 12

x = Variable(n)
y = Variable(n)
z = Variable(n-1)
i = Variable(n)
j = Variable(n)

a = NonNegative(n)
b = NonNegative(n)
c = NonNegative(n-1)

v = Variable(n)

x.value = np.ones((n,1))
y.value = np.ones((n,1))
z.value = np.ones((n-1,1))
i.value = np.ones((n,1))
j.value = np.ones((n,1))
a.value = np.ones((n,1))
b.value = np.ones((n,1))
c.value = np.ones((n-1,1))
v.value = np.ones((n,1))

constr = [i[0] == x[0], j[0] == y[0], i[n-1] == -I0, j[n-1] == -I0]
cost = 0
for k in range(n-1):
    cost += square(v[k]-v[k+1]-delta)
    #constr += [x[k]*a[k] == u0 - v[k]]#
    #constr += [y[k]*b[k] == v[k]]#
    constr += [z[k]*c[k] == v[k]-v[k+1]]
    constr += [i[k+1]== i[k]+x[k+1]]
    constr += [j[k+1] == j[k]+y[k+1]]
for k in range(n-2):
    constr += [x[k+1]+z[k] == y[k+1]+z[k+1]]
constr += [mul_elemwise(x,a) == u0 - v]
constr += [mul_elemwise(y,b) == v]
#constr += [x[n-1]*a[n-1] == u0 - v[n-1]]#
#constr += [y[n-1]*b[n-1] == v[n-1]]#
constr += [x[0] == y[0]+z[0]]
constr += [x[n-1]+z[n-2] == y[n-1]]

prob = Problem(Minimize(cost), constr)

for var in prob.variables():
    print var.name()
print find_minimal_sets(prob)

prob.solve(method = 'bcd', ep = 1e-2)

for k in range(n-1):
    print a[k].value, b[k].value, c[k].value
    print v[k].value-v[k+1].value
print a[n-1].value, b[n-1].value

