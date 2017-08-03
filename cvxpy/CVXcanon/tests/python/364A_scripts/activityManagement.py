from cvxpy import *
import numpy as np

# Create two scalar optimization variables.

A = np.array([ [1, 2, 0, 1], \
[0, 0, 3, 1], \
[0, 3, 1, 1], \
[2, 1, 2, 5], \
[1, 0, 3, 2] ])

A_star = hstack(A,A)

c_max = np.array([100] * 5)

p = np.array([3, 2, 7, 6])
p_disc = np.array([2, 1, 4, 2])

p_star = vstack(p, p_disc)

q = np.array([4, 10, 5, 10])

x_star = Variable( 8 )
constraints = [ A_star * x_star <= c_max, x_star >= 0 ]
for i in range(4):
	constraints.append( x_star[i] >= q[i] )

objective = Maximize(p_star.T * x_star)

prob = Problem(objective, constraints)
result = prob.solve() 

x = np.array( [0] * 4)
for i in range(4):
	x[i] = x_star.value[i] + x_star.value[4 + i]


pass #print "Optimal revenue:", result 
pass #print "Optimal activity levels:", x

average_rate = np.array([0] * 4)

for i in range(4):
	average_rate[i] = (x_star.value[i] * p_star.value[i] + x_star.value[i + 4] * p_star.value[i + 4]) / x[i]

pass #print "Average rate:", average_rate