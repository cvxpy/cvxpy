# Relax and round example for talk.
from __future__ import division
from cvxpy import *
import numpy

# def bool_vars(prob):
#     return [var for var in prob.variables() if var.boolean]

def cvx_relax(prob):
    new_constr = []
    for var in prob.variables():
        if getattr(var, 'boolean', False):
            new_constr += [0 <= var, var <= 1]
    return Problem(prob.objective,
                   prob.constraints + new_constr)

def round_and_fix(prob):
    prob.solve()
    new_constr = []
    for var in prob.variables():
        if getattr(var, 'boolean', False):
            new_constr += [var == numpy.round(var.value)]
    return Problem(prob.objective,
                   prob.constraints + new_constr)

def branch_and_bound(n, A, B, c):
    from Queue import PriorityQueue
    x = Variable(n)
    z = Variable(n)
    L = Parameter(n)
    U = Parameter(n)
    prob = Problem(Minimize(sum_squares(A*x + B*z - c)),
                   [L <= z, z <= U])
    visited = 0
    best_z = None
    f_best = numpy.inf
    nodes = PriorityQueue()
    nodes.put((numpy.inf, 0, numpy.zeros(n), numpy.ones(n), 0))
    while not nodes.empty():
        visited += 1
        # Evaluate the node with the lowest lower bound.
        _, _, L_val, U_val, idx = nodes.get()
        L.value = L_val
        U.value = U_val
        lower_bound = prob.solve()
        z_star = numpy.round(z.value)
        upper_bound = Problem(prob.objective, [z == z_star]).solve()
        f_best = min(f_best, upper_bound)
        if upper_bound == f_best:
            best_z = z_star
        # Add new nodes if not at a leaf and the branch cannot be pruned.
        if idx < n and lower_bound < f_best:
            for i in [0, 1]:
                L_val[idx] = U_val[idx] = i
                nodes.put((lower_bound, i, L_val.copy(), U_val.copy(), idx + 1))

    #print("Nodes visited: %s out of %s" % (visited, 2**(n+1)-1))
    return f_best, best_z

# def round_and_fix2(prob, thresh):
#     prob.solve()
#     new_constr = []
#     for var in bool_vars(prob):
#         new_constr += [var == (var.value > thresh)]
#     return Problem(prob.objective, prob.constraints + new_constr)

# def round_and_fix3(prob, thresh):
#     prob.solve()
#     new_constr = []
#     for var in bool_vars(prob):
#         print var.value
#         new_constr += [(var.value > 1 - thresh ) <= var,
#                        var <= ~(var.value <= thresh)]
#     return Problem(prob.objective, prob.constraints + new_constr)

numpy.random.seed(1)

# Min sum_squares(A*x + B*z - c)
# z boolean.
def example(n, get_vals=False):
    print "n = %d #################" % n
    m = 2*n
    A = numpy.matrix(numpy.random.randn(m, n))
    B = numpy.matrix(numpy.random.randn(m, n))
    sltn = (numpy.random.randn(n, 1),
            numpy.random.randint(2, size=(n, 1)))
    noise = numpy.random.normal(size=(m, 1))
    c = A.dot(sltn[0]) + B.dot(sltn[1]) + noise

    x = Variable(n)
    #x.boolean = False
    z = Variable(n)
    z.boolean = True

    obj = sum_squares(A*x + B*z - c)
    prob = Problem(Minimize(obj))
    relaxation = cvx_relax(prob)
    print "relaxation", relaxation.solve()
    rel_z = z.value
    rounded = round_and_fix(relaxation)
    rounded.solve()
    print "relax and round", rounded.value
    truth, true_z = branch_and_bound(n, A, B, c)
    print "true optimum", truth
    if get_vals:
        return (rel_z, z.value, true_z)
    return (relaxation.value, rounded.value, truth)

# Plot relaxation z_star.
import matplotlib.pyplot as plt
n = 20
vals = range(1, n+1)
relaxed, rounded, truth = map(numpy.asarray, example(n, True))
plt.figure(figsize=(6,4))
plt.plot(vals, relaxed, 'ro')
plt.axhline(y=0.5,color='k',ls='dashed')
plt.xlabel(r'$i$')
plt.ylabel(r'$z^\mathrm{rel}_i$')
plt.show()

# Plot optimal values.
import matplotlib.pyplot as plt
relaxed = []
rounded = []
truth = []
vals = range(1, 36)
for n in vals:
    results = example(n)
    results = map(lambda x: numpy.around(x, 3), results)
    relaxed.append(results[0])
    rounded.append(results[1])
    truth.append(results[2])

plt.figure(figsize=(6,4))
plt.plot(vals, rounded, vals, truth, vals, relaxed)
plt.xlabel("n")
plt.ylabel("Objective value")
plt.legend(["Relax and round value", "Global optimum", "Lower bound"], loc=2)
plt.show()

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

# # Warehouse operation.
# #  http://web.mit.edu/15.053/www/AMP-Chapter-09.pdf
# # cost per unit from warehouse i to customer j
# # cost for warehouse being used
# # fixed customer demand
# m = 100 # number of customers.
# n = 50 # number of warehouses.

# numpy.random.seed(1)
# C = numpy.random.random((n, m))
# f = numpy.random.random((n, 1))
# d = numpy.random.random((m, 1))
# X = Variable(n, m)
# y = Variable(n)

# # Annotate variables.
# X.boolean = False
# y.boolean = True
# demand = [sum_entries(X[:, j]) == d[j] for j in range(m)]
# valid = [sum_entries(X[i, :]) <= y[i]*d.sum() for i in range(n)]
# obj = sum_entries(mul_elemwise(C, X)) + f.T*y
# prob = Problem(Minimize(obj),
#                [X >= 0, sum_entries(y) >= 3*n/4] + demand + valid)

# relaxation = cvx_relax(prob)
# print relaxation.solve()
# rounded = round_and_fix(relaxation)
# # rounded = relaxation
# # K = 4
# # for i in range(K):
# #     print i
# #     rounded = round_and_fix3(rounded, i/(2*K))
# #     print y.value.sum()
# print rounded.solve()
# print rounded.status
# print y.value.sum()
# # print numpy.around(X.value, 2)
# # print numpy.around(y.value, 2)
