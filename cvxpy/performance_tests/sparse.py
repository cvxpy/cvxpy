import cvxpy as cp
import cvxpy.settings as s
import itertools
from cvxopt import spmatrix
import cvxopt
import numpy as np
import time

import cProfile
cProfile.run("""
import cvxpy as cp
n = 2000
A = cp.Variable(n*n, 1)
obj = cp.Minimize(cp.norm(A, 'fro'))
p = cp.Problem(obj, [A >= 2])
result = p.solve(verbose=True)
print result
""")

# m = 100
# n = 1000
# prob = 0.999

# a_arr = np.random.random((m, n))
# a_arr[a_arr < prob] = 0

# a_arr_sp = spmatrix(a_arr[a_arr.nonzero()[0],
#                           a_arr.nonzero()[1]],
#                     a_arr.nonzero()[0],
#                     a_arr.nonzero()[1],
#                     size=(m, n))

# W = cp.Variable(n, n)
# constraints = []

# constraints.extend( [W[i,i] == 0 for i in range(n)] )
# constraints.append(W >= 0)
# lam = 8
# beta = 0.5
# loss = cp.sum(a_arr_sp - a_arr_sp*W)
# l2_reg = 0.5*beta*cp.square(cp.norm(W))
# l1_reg = lam*cp.sum(W)
# obj = cp.Minimize(loss + l2_reg + l1_reg)
# # TODO No constraints, get error.
# p = cp.Problem(obj, constraints)

# import cProfile
# cProfile.run('p.solve()')


# objective, constr_map, dims = p.canonicalize()

# all_ineq = itertools.chain(constr_map[s.EQ], constr_map[s.INEQ])
# var_offsets, x_length = p._get_var_offsets(objective, all_ineq)

# c, obj_offset = p._constr_matrix([objective], var_offsets, x_length,
#                                  p._DENSE_INTF,
#                                  p._DENSE_INTF)
# A, b = p._constr_matrix(constr_map[s.EQ], var_offsets, x_length,
#                            p._SPARSE_INTF, p._DENSE_INTF)

# G, h = p._constr_matrix(constr_map[s.INEQ], var_offsets, x_length,
#                            p._SPARSE_INTF, p._DENSE_INTF)

# print len(constr_map[s.EQ])
# print len(constr_map[s.INEQ])

# cProfile.run("""
# G, h = p._constr_matrix(constr_map[s.INEQ], var_offsets, x_length,
#                            p._SPARSE_INTF, p._DENSE_INTF)
# """)
# import numbers

# aff_expressions = constr_map[s.INEQ]
# matrix_intf, vec_intf = p._SPARSE_INTF, p._DENSE_INTF

# expr_offsets = {}
# vert_offset = 0
# for aff_exp in aff_expressions:
#     expr_offsets[str(aff_exp)] = vert_offset
#     vert_offset += aff_exp.size[0]*aff_exp.size[1]

# #rows = sum([aff.size[0] * aff.size[1] for aff in aff_expressions])
# rows = vert_offset
# cols = x_length
# #const_vec = vec_intf.zeros(rows, 1)
# vert_offset = 0

# def carrier(expr_offsets, var_offsets):
#     def f(aff_exp):
#         V, I, J = [], [], []
#         vert_offset = expr_offsets[str(aff_exp)]
#         coefficients = aff_exp.coefficients()
#         for var, blocks in coefficients.items():
#             # Constant is not in var_offsets.
#             horiz_offset = var_offsets.get(var)
#             for col, block in enumerate(blocks):
#                 vert_start = vert_offset + col*aff_exp.size[0]
#                 vert_end = vert_start + aff_exp.size[0]
#                 if var is s.CONSTANT:
#                     pass
#                     #const_vec[vert_start:vert_end, :] = block
#                 else:
#                     if isinstance(block, numbers.Number):
#                         V.append(block)
#                         I.append(vert_start)
#                         J.append(horiz_offset)
#                     else: # Block is a matrix or spmatrix.
#                         if isinstance(block, cvxopt.matrix):
#                             block = cvxopt.sparse(block)
#                         V.extend(block.V)
#                         I.extend(block.I + vert_start)
#                         J.extend(block.J + horiz_offset)
#         return (V, I, J)
#     return f

# f = carrier(expr_offsets, var_offsets)

# from multiprocessing import Pool
# p = Pool(1)
# result = p.map(f, aff_expressions)
# V, I, J = [], [], []
# for v, i, j in result:
#     V.extend(v)
#     I.extend(i)
#     J.extend(j)

# #[item for sublist in l for item in sublist]
# # Create the constraints matrix.
# if len(V) > 0:
#     matrix = cvxopt.spmatrix(V, I, J, (rows, cols), tc='d')
#     # Convert the constraints matrix to the correct type.
#     matrix = matrix_intf.const_to_matrix(matrix, convert_scalars=True)
# else: # Empty matrix.
#     matrix = matrix_intf.zeros(rows, cols)
