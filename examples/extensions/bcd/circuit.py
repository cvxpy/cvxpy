__author__ = 'Xinyue'

from cvxpy import *
from bcd import bcd
import numpy as np

n = 6
m = 4
k = 5

I = Variable(n)
#U = NonNegative(m)
R = NonNegative(k)

cost = square(R[4]*I[5]-1)
constr = [ I[0] == 10, I[0] == I[1]+I[2], I[1] == I[5]+I[3], I[4] == I[2]+I[5]]
constr += [R>=0.001, R<=100]
constr += [R[0]*I[1]+R[4]*I[5]-R[1]*I[2] == 0, R[4]*I[5]+R[3]*I[4]-R[2]*I[3] == 0]
prob = Problem(Minimize(cost), constr)
prob.solve(method = 'bcd', solver = 'SCS')
print "===="
print R[4].value*I[5].value
print R.value
print I.value

