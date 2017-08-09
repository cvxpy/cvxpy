from cvxpy import *
import numpy
import time

ANSWERS = []
TIME = 0 
x = Variable(2)

constraints = [ 2 * x[0] + x[1] >= 1, x[0] + 3 * x[1] >= 1, \
x[0] >= 0, x[1] >= 0 ]

objectives  =   [x[0] + x[1], -x[0] - x[1], x[0],\
max_elemwise( x[0], x[1] ), x[0] ** 2 + 9 * x[1] ** 2 ]    

for n, obj in enumerate(objectives):
	prob = Problem(Minimize(obj), constraints)
	tic = time.time()
	ANSWERS.append(prob.solve())
	toc = time.time()
	TIME += toc - tic
	pass #print "Solution to objective " + str(n)
	pass #print x.value
	pass #print obj.value
	pass #print