import cvxpy as cp
import cvxpy.settings as s
import itertools
from cvxopt import spmatrix
import numpy as np
import time

m = 100
n = 4503
prob = 0.999

a_arr = np.random.random((m, n))
a_arr[a_arr < prob] = 0

a_arr_sp = spmatrix(a_arr[a_arr.nonzero()[0],
						  a_arr.nonzero()[1]],
					a_arr.nonzero()[0],
					a_arr.nonzero()[1],
					size=(m, n))

W = cp.Variable(n, 5)
constraints = []

constraints.append(W[0,0] == 0)
constraints.append(W[:,0] >= 0)
lam = 8
beta = 0.5
loss = cp.sum(a_arr_sp[:,0] - a_arr_sp*W[:,0])
l2_reg = 0.5*beta*cp.sum(cp.square(W[:,0]))
l1_reg = lam*cp.sum(W[:,0])
obj = cp.Minimize(l2_reg)
# TODO No constraints, get error.
p = cp.Problem(obj, constraints)

import cProfile
#cProfile.run('p.solve()')
objective, constr_map, dims = p.canonicalize()

all_ineq = itertools.chain(constr_map[s.EQ], constr_map[s.INEQ])
var_offsets, x_length = p._get_var_offsets(objective, all_ineq)

c, obj_offset = p._constr_matrix([objective], var_offsets, x_length,
                                 p._DENSE_INTF,
                                 p._DENSE_INTF)
A, b = p._constr_matrix(constr_map[s.EQ], var_offsets, x_length,
                           p._SPARSE_INTF, p._DENSE_INTF)
G, h = p._constr_matrix(constr_map[s.INEQ], var_offsets, x_length,
                           p._SPARSE_INTF, p._DENSE_INTF)

print len(constr_map[s.EQ])
print len(constr_map[s.INEQ])
cProfile.run("""
G, h = p._constr_matrix(constr_map[s.INEQ], var_offsets, x_length,
                           p._SPARSE_INTF, p._DENSE_INTF)
""")