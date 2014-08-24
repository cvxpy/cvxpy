# Relax and round example for talk.
from __future__ import division
from cvxpy import *
import numpy

def bool_vars(prob):
    return [var for var in prob.variables() if var.boolean]

def cvx_relax(prob):
    new_constr = []
    for var in bool_vars(prob):
        new_constr += [0 <= var, var <= 1]
    return Problem(prob.objective, prob.constraints + new_constr)

def round_and_fix(prob):
    prob.solve()
    new_constr = []
    for var in bool_vars(prob):
        new_constr += [var == numpy.round(var.value)]
    return Problem(prob.objective, prob.constraints + new_constr)


def round_and_fix2(prob, thresh):
    prob.solve()
    new_constr = []
    for var in bool_vars(prob):
        new_constr += [var == (var.value > thresh)]
    return Problem(prob.objective, prob.constraints + new_constr)

def round_and_fix3(prob, thresh):
    prob.solve()
    new_constr = []
    for var in bool_vars(prob):
        print var.value
        new_constr += [(var.value > 1 - thresh ) <= var,
                       var <= ~(var.value <= thresh)]
    return Problem(prob.objective, prob.constraints + new_constr)

def all_bools(prob):
    possibs = []

numpy.random.seed(1)

# Min sum_squares(A*x + B*z - c)
# z boolean.
#

# m = 10
# n = 8
# nnz = 5
# A = numpy.random.randn(m, n)
# solution = numpy.random.randint(2, size=(n, 1))
# b = A.dot(solution)
# x = Variable(n)
# y = Variable(n)
# x.boolean = False
# y.boolean = True
# U = 100
# L = -100
# obj = sum_squares(A*x - b)
# constraints = [L*y <= x, x <= U*y,
#                sum_entries(y) <= nnz]
# prob = Problem(Minimize(obj), constraints)
# relaxation = cvx_relax(prob)
# print relaxation.solve()
# rounded = relaxation
# K = 4
# for i in range(K+1):
#     rounded = round_and_fix3(rounded, i/(2*K))
# print rounded.solve()
# print numpy.around(x.value, 2)
# print numpy.around(y.value, 2)

# Warehouse operation.
#  http://web.mit.edu/15.053/www/AMP-Chapter-09.pdf
# cost per unit from warehouse i to customer j
# cost for warehouse being used
# fixed customer demand
m = 100 # number of customers.
n = 50 # number of warehouses.

numpy.random.seed(1)
C = numpy.random.random((n, m))
f = numpy.random.random((n, 1))
d = numpy.random.random((m, 1))
X = Variable(n, m)
y = Variable(n)

# Annotate variables.
X.boolean = False
y.boolean = True
demand = [sum_entries(X[:, j]) == d[j] for j in range(m)]
valid = [sum_entries(X[i, :]) <= y[i]*d.sum() for i in range(n)]
obj = sum_entries(mul_elemwise(C, X)) + f.T*y
prob = Problem(Minimize(obj),
               [X >= 0, sum_entries(y) >= 3*n/4] + demand + valid)

relaxation = cvx_relax(prob)
print relaxation.solve()
rounded = round_and_fix(relaxation)
# rounded = relaxation
# K = 4
# for i in range(K):
#     print i
#     rounded = round_and_fix3(rounded, i/(2*K))
#     print y.value.sum()
print rounded.solve()
print rounded.status
print y.value.sum()
# print numpy.around(X.value, 2)
# print numpy.around(y.value, 2)
