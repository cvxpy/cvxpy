import pdb

import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp

y = np.array([20.79, 20.79, 22.40, 22.67, 23.15, 23.35, 23.89,
              23.99, 24.02, 24.01, 25.14, 26.57, 28.49, 27.76,
              29.04, 29.88, 30.06])
x = np.array([194.5, 194.3, 197.9, 198.4, 199.4, 199.9,
              200.9, 201.1, 201.4, 201.3, 203.6, 204.6,
              209.5, 208.6, 210.7, 211.9, 212.2])
x = x - np.mean(x)
n = len(y)

# most general EML
w = cp.Variable(n, nonneg=True)
beta0 = cp.Variable()
beta1 = cp.Variable()
constraints = [cp.sum(w) == 1, cp.sum(cp.multiply(w, y - beta0 - cp.multiply(beta1, x))) == 0]
obj = cp.Maximize(cp.sum(cp.log(w)))
prob = cp.Problem(obj, constraints)
prob.solve(nlp=True, verbose=True, derivative_test='none', solver='IPOPT')

print("optimal beta:                      ", beta0.value, beta1.value)
print("standard regression cvoefficients: ", np.mean(y), x.T @ y / (x.T @ x))

# generate plot of log empirical likelihood as a function of slope and intercept
def log_eml(beta0_val, beta1_val):
    w_val = cp.Variable(n, nonneg=True)
    constraints = [cp.sum(w_val) == 1,
                   cp.sum(cp.multiply(w_val, y - beta0_val - cp.multiply(beta1_val, x))) == 0]
    obj = cp.Maximize(cp.sum(cp.log(w_val)))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, verbose=True)
    return prob.value

slopes = np.linspace(0.4, 0.7, 50)
intercepts = np.linspace(24.6, 25.8, 50)
log_emls = np.zeros((len(intercepts), len(slopes)))
for i, b0 in enumerate(intercepts):
    for j, b1 in enumerate(slopes):
        log_emls[i, j] = log_eml(b0, b1)

pdb.set_trace()

# do contour plot
B1, B0 = np.meshgrid(slopes, intercepts)
plt.contour(B0, B1, log_emls, levels=30)
plt.xlabel('slope')
plt.ylabel('intercept')
plt.title('Log Empirical Likelihood Contours')
plt.colorbar(label='Log Empirical Likelihood')
plt.legend()
plt.show()