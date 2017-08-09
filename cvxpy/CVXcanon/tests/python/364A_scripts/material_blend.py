import numpy as np
from cvxpy import *
import time

ANSWERS = []
TIME = 0
m = 4 # the number of raw materials
n = 2 # the number of blended materials
q = 3 # the number of constituents
# the ith column of C is c_i
C = np.array([[ .9,  .8, .7, .6],
              [.08, .12, .2, .2],
              [.02, .08, .1, .2]])

# bounds on the blended product concentration
c_min = np.array([[.85, 0.65],
                    [  0,  0],
                    [  0,  0]])
c_max = np.array([[ 1,  1 ],
                  [.1, .18],
                  [.05, .17]])

FTilde = np.array([10, 10]) # limit on the flow rate of the blended material
F = np.array([7, 2, 6, 3]) # availibility of raw materials

p = np.array([15, 13, 11, 8]) # price of raw materials
pTilde = np.array([21, 18]) # price of the blended material


f_tilde = Variable(n)
f = Variable(m,n)
M = Variable(n,q)

constraints = [f >= 0, f_tilde <= FTilde]
for i in range(m):
	constraints.append(sum(f[i,:]) <= F[i])

for i in range(n):
	to_add = [f[k,i] * C[:,k] for k in range(m)]
	constraints.append(M[i,:].T == sum(to_add))
	constraints.append(c_min[:,i]*f_tilde[i]  <= M[i,:].T )
	constraints.append(c_max[:,i]*f_tilde[i]  >= M[i,:].T )
	constraints.append(sum( M[i,:]) == f_tilde[i] )

to_add = [-f[j,i] * p[j] for i in range(n) for j in range(m)]
obj = sum(to_add)
obj += sum([f_tilde[i] * pTilde[i] for i in range(n)])

objective = Maximize(obj)
problem = Problem(objective, constraints)
tic = time.time()
val = problem.solve()
toc = time.time()
TIME += toc - tic
ANSWERS.append(val)
pass #print val
pass #print f.value
pass #print f_tilde.value
for i in range(n):
	pass #print M.value[i,:] / f_tilde[i].value