from cvxpy import *
import numpy as np

# x = Variable(2)
# x = Variable(2)
# y = Variable(2)
# print Problem(Minimize(0), [x + y <= [1, 0]]).solve()

x = Variable(2)
constraints = [ 2 * x[0] + x[1] >= 1, x[0] + 3 * x[1] >= 1, \
x[0] >= 0, x[1] >= 0 ]

objectives  =   [x[0] + x[1], -x[0] - x[1], x[0],\
max_elemwise( x[0], x[1] ), x[0] ** 2 + 9 * x[1] ** 2 ]

for n, obj in enumerate(objectives):
	settings.USE_CVXCANON = False

	print "\n\nNormal Solution:\n"
	prob = Problem(Minimize(obj), constraints)
	prob.solve()
	print "Solution to objective " + str(n)
	print x.value
	print obj.value

	settings.USE_CVXCANON = True
	
	print "\n\nCVXCANON Solution:\n"
	prob = Problem(Minimize(obj), constraints)
	prob.solve()
	print "Solution to objective " + str(n)
	print x.value
	print obj.value