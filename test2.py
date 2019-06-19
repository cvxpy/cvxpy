import numpy as np
import cvxpy as cp
from cvxpy import Variable
from cvxpy.atoms.affine.trace import trace

Z=Variable((2,2),hermitian=True)
constraints=[trace(cp.real(Z))==1]
obj=cp.Minimize(0)
prob=cp.Problem(obj,constraints)
prob.solve()
