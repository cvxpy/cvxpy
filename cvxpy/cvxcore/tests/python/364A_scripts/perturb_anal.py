import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt
import time

ANSWERS = []
TIME = 0

A = np.zeros( (2,2) )
A[0][0] = 1
A[1][1] = 2
A[0][1] = A[1][0] = .5

x = Variable(2)
obj = Minimize( quad_form(x, A) - x[0])
u_1 = Parameter()
u_2 = Parameter()

constraints = [x[0] + 2*x[1] <= u_1,\
x[0] - 4 * x[1] <= u_2,\
5 * x[0] + 76 * x[1] <= 1]

u_1.value = -2
u_2.value = -3

prob = Problem(obj, constraints)

tic = time.time()
p_star = prob.solve()
toc = time.time()
TIME += toc - tic
ANSWERS.append(p_star)
print "Optimal value,", p_star

print "x,", x.value

print "Lambda 1", constraints[0].dual_value
print "Lambda 2", constraints[1].dual_value
print "Lambda 3", constraints[2].dual_value

delta_1 = [0, -.1, .1]
delta_2 = [0, -.1, .1]

for d1 in delta_1:
	for d2 in delta_2:
		u_1.value = -2 + d1 
		u_2.value = -3 + d2
		val = prob.solve()
		ANSWERS.append(val)
		print d1, d2, val
