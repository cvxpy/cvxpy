import numpy as np
from cvxpy import *
import math


ANSWERS = []
A = np.array(np.mat('-1 -1 0 0 0;\
	0 1 -1 -1 0;\
	0 0 0 1 1'))

I = Variable(5)
I_entr = Variable(2)
f_0 = -entr(I_entr[0])-I[2] - 26.0 * log(I[2])
f_0 += -entr(I_entr[1])-I[3]  - 26.0 * log(I[3])
constraints = []
constraints = [I_entr[0] == I[2] - 26.0, I_entr[1] == I[3] - 26.0 ]
constraints.append(A*I == 0)
objective = Minimize(f_0)
prob = Problem(objective, constraints)
val = prob.solve()

ANSWERS.append(val)
v = [None] * 5
v[0] = 1000*I.value[0]+I.value[0]
v[1] = 1000*I.value[1] 
v[4] = 100*I.value[4] 
v[3] =    math.log(1+ I.value[3]/26.0) 
v[2] =    math.log(1 + I.value[2]/26.0) 

v = np.mat(v)
e = np.linalg.pinv(A.T)*v.T

pass #print e == A * v.T

pass #print e
pass #print v