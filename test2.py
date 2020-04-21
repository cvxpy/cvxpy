import cvxpy as cp
import numpy as np

Pref = np.random.randint(-1, 1, (28, 28))

p = Pref.shape[0]
size = 4
g = int(p / size)

# xb = cp.Variable((p, g), boolean=True)
xb = cp.Variable((p,g))
x = xb
constraints = [cp.sum(x, axis=1) == 1, cp.sum(x, axis=0) == size]

cost_ = cp.sum(Pref * x)
cost_
cost = cp.Maximize(cost_)
prob = cp.Problem(cost, constraints)
status = prob.solve()
print(status)

if status == "optimal":
    print(x.value)
