import cvxpy as cp
import numpy as np
import scipy.linalg


n = 4
L = np.random.randn(n, n)
P = L.T @ L
P_sqrt = cp.Parameter((n, n))
x = cp.Variable((n, 1))
quad_form = cp.sum_squares(P_sqrt @ x)
P_sqrt.value = scipy.linalg.sqrtm(P)
print(quad_form.is_dpp())
