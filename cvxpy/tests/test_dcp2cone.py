import cvxpy as cvx
import numpy.random as r
import numpy as np
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone

n = 5
x = cvx.Variable(n)
y = cvx.Variable()
A = r.rand(n,n)
b = r.rand(n,1)

c = [cvx.abs(A*x + b) <= 2, cvx.abs(y) + x[0] <= 1]
cvx.Minimize(x[0])
prob = cvx.Problem(cvx.Minimize(x[0] + x[1] + y), c)
 
d2c = Dcp2Cone()
d2c.accepts(prob)
new_prob = d2c.apply(prob)

print(prob)
print('\n\n')
print(new_prob)

prob.solve()
new_prob.solve()

print(prob.value)
print(new_prob.value)


