from cvxpy import *
import numpy

x = Variable(2)

constraints = [ 2 * x[0] + x[1] >= 1, x[0] + 3 * x[1] >= 1, \
x[0] >= 0, x[1] >= 0 ]

objectives  =   [x[0] + x[1]]    

for n, obj in enumerate(objectives):
	prob = Problem(Minimize(obj), constraints)
	prob.solve()
	print "Solution to objective " + str(n)
	print x.value
	print obj.value
	print