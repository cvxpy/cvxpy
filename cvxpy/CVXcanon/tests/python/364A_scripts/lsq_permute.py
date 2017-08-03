import numpy as np
from cvxpy import *
import matplotlib.pyplot as pyplot
import heapq
import time

settings.USE_CVXCANON = True
ANSWERS = []
TIME = 0
np.random.seed(0)
m=100
k=40 # max # permuted measurements
n=20
A=10 * np.random.randn(m,n)
x_true=np.random.randn(n,1) # true x value
y_true = A.dot(x_true) + np.random.randn(m,1)

# build permuted indices
perm_idxs=np.random.permutation(m)
perm_idxs=np.sort(perm_idxs[:k])
temp_perm=np.random.permutation(k)
new_pos=np.zeros(k)
for i in range(k):
  new_pos[i] = perm_idxs[temp_perm[i]]
new_pos = new_pos.astype(int)

# true permutation matrix
P=np.identity(m)
P[perm_idxs,:]=P[new_pos,:]
true_perm=[]

for i in range(k):
  if perm_idxs[i] != new_pos[i]:
    true_perm = np.append(true_perm, perm_idxs[i])

y = P.dot(y_true)
new_pos = None



P_fixed = np.identity(m)
x_fixed = None


def optimizeP(A, x, y):
	P = np.identity(m)
	Ax = A.dot(x)
	
	Ax_largest = heapq.nlargest(m, range(len(Ax)), Ax.take)
	y_largest = heapq.nlargest(m, range(len(y)), y.take)


	for i in range(m):
		P[ y_largest[i], y_largest[i] ] = 0
		P[ y_largest[i], Ax_largest[i] ] = 1

	return P



def numPermuted(P):
	result = 0
	for i in range(m):
		if P[i,i] != 1:
			result += 1
	return result		

firstIter = None

for _ in range(20):

	x = Variable(n)
	objective = Minimize( norm( A*x - P_fixed.T.dot( y)) )
	constraints = []
	prob = Problem(objective, constraints)
	tic = time.time()
	result = prob.solve()
	toc = time.time()
	TIME += toc - tic
	ANSWERS.append(result)
	if firstIter is None:
		firstIter = result
	x_fixed = x.value

	P_fixed = optimizeP(A, x_fixed, y)


print "Num permuted the same as k: ",  numPermuted(P) == k 
print "Final objective", result
print "P = I error", firstIter
