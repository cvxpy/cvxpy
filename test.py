import numpy as np

import cvxpy as cp



s = cp.Variable((2, 2))

obj= cp.Maximize(cp.minimum(s[0,1], 10))



const = [s>>0,cp.diag(s)==np.ones(2)]



prob = cp.Problem(obj, const)

r = prob.solve(verbose=True)

s = s.value

s2 = (s+s.T)/2

print(const[0].residual)
print("value", r)

print("s", s)

print("eigs",np.linalg.eig(s))
print("eigs",np.linalg.eig(s2)[0])
