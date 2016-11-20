import cvxpy as cvx
import numpy.random as r
#from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from dcp2cone import Dcp2Cone

n = 5
x = cvx.Variable(n)
A = r.rand(n,n)
b = r.rand(n,1)

c = [abs(A*x + b) <= 2]
prob = cvx.Problem(cvx.Minimize(abs(x)), c)
 
d2c = Dcp2Cone()
d2c.accepts(prob)
d2c.apply(prob)


