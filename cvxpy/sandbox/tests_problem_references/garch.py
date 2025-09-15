import os
import pdb

import numpy as np
import pandas as pd

import cvxpy as cp

print("CWD:", os.getcwd())


df = pd.read_csv("labML.csv", index_col=0, header=1)
S = df.to_numpy().reshape(-1)
#S = S[0:200]
r = np.log(S[1:] / S[:-1])
T = len(S)
dt = 1 / 252


# DEUBG idea: fix some variables and solve for the others. When does it start to fail?

# Formulation one with v = sigma2 
#beta = cp.Variable((3, ), nonneg=True)


beta0 = cp.Variable((1, ), nonneg=True, name="b0")
beta1 = cp.Variable((1, ), nonneg=True, name="b1")
beta2 = cp.Variable((1, ), nonneg=True, name="b2")
nu = cp.Variable((1, ), name="nu", bounds = [-0.2, 0.2])
v = cp.Variable((T - 1, ), nonneg=True, name="v")
#nu = cp.Variable((1, ), name="nu", bounds = [-0.3, 0.3])
#nu = cp.Variable((1, ), name="nu")


term1 = cp.sum(cp.log(2 * np.pi * dt * v))
term2 = cp.sum(cp.square(r - nu * dt) / (v * dt))
obj = term1 + term2
constraints = [v[1:] == beta0 + beta1 * v[:-1] + beta2 * (r[:-1] ** 2 / dt),
               beta1 + beta2 <= 1]
problem = cp.Problem(cp.Minimize(obj), constraints)


# initialization
beta0.value = np.array([0]) 
beta1.value = np.array([0.931])
beta2.value = np.array([0.061]) 
nu.value = np.array([0.051]) 
#v0 = np.zeros((T - 1, ))
#v0[0] = np.std(r) ** 2 / dt
#for t in range(1, T - 1):
#    v0[t] = beta0.value[0] + beta1.value[0] * v0[t - 1] + beta2.value[0] * (r[t - 1] ** 2 / dt)
#v.value = v0
#v.value = np.ones(v.shape)


problem.solve(solver=cp.IPOPT, nlp=True)

loglikelihood = -0.5 * problem.value
manual_log_likelihood = -0.5 * (np.sum(np.log(2 * np.pi * dt * v.value)) + \
                                np.sum((r - nu.value * dt) ** 2 / (v.value * dt)))
manual_primal_residual = np.linalg.norm(v.value[1:] - (beta0.value + beta1.value * v.value[:-1] + \
                                                       beta2.value * (r[:-1] ** 2 / dt)))

print("loglikelihood: ", loglikelihood)
print("manual_log_likelihood: ", manual_log_likelihood)
print("manual_primal_residual: ", manual_primal_residual)
print("beta0.value: ", beta0.value)
print("beta1.value: ", beta1.value)
print("beta2.value: ", beta2.value)
print("nu.value: ", nu.value)
#print("v.value: ", v.value)

pdb.set_trace()

pdb.set_trace()