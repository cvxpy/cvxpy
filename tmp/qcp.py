import cvxpy as cp
import numpy as np


# problem data
n = 3
# transitions
W = np.random.random((n + 1, n + 1))
W /= W.sum(axis=1)[:,None]
W[-1, :] = np.zeros(n  + 1)
W[-1, -1] = 1.
# visits
Q = np.linalg.inv(np.eye(n) - W[:n,:n])
# entry vec
sigma = np.random.rand(n)
sigma = sigma / sigma.sum()
# time
times = np.random.rand(n)
# max supply
max_supply_s = 10.
max_supply_w = 2.

# price
price = cp.Variable((n,), pos=True)
price.value = np.zeros(n)
# revenue
rev = sigma.dot(Q) * price

# shift
shift = sigma.dot(Q).dot(times)

# supply func
def F_w(wage):
    lam =50
    k = 1
    return 1 - (cp.exp(-(wage/lam)**k))
# wage 
wage = rev / shift
# supply based on wage
wage_supply_s = max_supply_s * F_w(wage)
wage_supply_w = max_supply_w * F_w(wage)
assert wage_supply_s.is_dqcp() and wage_supply_w.is_dqcp()

# demand func
def F_p(p):
    lam = 10
    k = 1
    return (cp.exp(-(p/lam)**k))

# max demands
max_demand_s = np.random.rand(n) * 10
max_demand_w = max_demand_s * 0.2

# demand constants
demand_constant_s = max_demand_s / sigma.dot(Q)
demand_constant_w = max_demand_w / sigma.dot(Q)

# scaled demands
scaled_demand_s = []
for i in range(n):
    scaled_demand_s.append(
        demand_constant_s[i] * F_p(price[i])
    )
    
scaled_demand_w = []
for i in range(n):
    scaled_demand_w.append(
        demand_constant_w[i] * F_p(price[i])
    )
    
scaled_demand_tot = []
for i in range(n):
    scaled_demand_tot.append(
        (demand_constant_s[i] + demand_constant_w[i]) * F_p(price[i])
    )
# s constraints
h = cp.minimum(
    wage_supply_s,
    cp.minimum(*scaled_demand_s)
)
problem = cp.Problem(cp.Maximize(h))
print(problem)
problem.solve(qcp=True, verbose=True)
print("optimal price: ", price.value)
