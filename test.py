import cvxpy as cp
import numpy as np

m = 15
n = 10
p = 5
np.random.seed(1)
P = np.random.randn(n, n)
P = P.T @ P
q = np.random.randn(n)
s = np.random.randn(n)

# define the optimization problem with the 2nd constraint as a quad_form constraint
x = cp.Variable(n)
prob = cp.Problem(cp.Maximize(q.T @ x - (1/2)*cp.quad_form(x, P)),
                 [cp.norm(x, 1) <= 1.0,
                  cp.quad_form(x, P) <= 10,   # quad form constraint
                  cp.abs(x) <= 0.01
                  ])
prob.solve()

# access quad_form.expr.grad
a = prob.constraints[1].expr.grad
