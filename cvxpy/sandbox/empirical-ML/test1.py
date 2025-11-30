import pdb

import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp

np.random.seed(0)
n = 20

x = np.array([[28, -44, 29, 30, 26, 27, 22, 23, 33, 16, 24, 29,
              24, 40, 21, 31, 34, -2, 25, 19]]).T

x = x - np.mean(x)

w = cp.Variable((n, 1), nonneg=True)
sigma2 = cp.Variable((), nonneg=True)
mu = cp.Variable(())
cost = cp.sum(cp.log(w))
constr = [cp.sum(w) == 1, w.T @ x == mu, cp.sum(cp.multiply(w, (x - mu) ** 2)) == sigma2]  
 # this crashes if w and x are 0-dimensional
mu.value = np.mean(x) # initialization
prob = cp.Problem(cp.Maximize(cost), constr)
prob.solve(nlp=True, verbose=True)

def R_fixed_sigma2(fixed_sigma2, x):
    w = cp.Variable((n, 1), nonneg=True)
    mu = cp.Variable(())
    cost = cp.sum(cp.log(w))
    constr = [cp.sum(w) == 1, w.T @ x == mu, 
              cp.sum(cp.multiply(w, (x - mu) ** 2)) == fixed_sigma2]   
    prob = cp.Problem(cp.Maximize(cost), constr)
    prob.solve(nlp=True, verbose=True, derivative_test='none')
    return mu.value

mu_opt, sigma2_opt = mu.value, sigma2.value
ipopt_value = prob.value
print("Optimal mu: ", mu_opt)
print("Optimal sigma^2: ", sigma2_opt)
print("mean of data: ", np.mean(x))
print("variance of data: ", np.std(x) ** 2)

#import pdb 
#pdb.set_trace()

def R(mu, sigma2, x):
    w = cp.Variable((n, 1), nonneg=True)
    x = x.reshape((n, 1))
    cost = cp.sum(cp.log(w))
    constr = [cp.sum(w) == 1,
              w.T @ x == mu,
              cp.sum(cp.multiply(w, (x - mu) ** 2)) == sigma2]
    prob = cp.Problem(cp.Maximize(cost), constr)
    prob.solve(solver=cp.MOSEK, verbose=False)
    return prob.value


# make contour plot
sigma_vals = np.logspace(-1, 2, 40)


mu_optimized = np.zeros(len(sigma_vals))
for j, sigma in enumerate(sigma_vals):
    mu_optimized[j] = R_fixed_sigma2(sigma**2, x)

mu_vals = np.linspace(-10, 10, 40)
R_vals = np.zeros((len(mu_vals), len(sigma_vals)))
for i, mu in enumerate(mu_vals):
    for j, sigma in enumerate(sigma_vals):
        R_vals[i, j] = R(mu, sigma**2, x)



pdb.set_trace()

print("largest value grid search: ", np.max(R_vals))
print("ipopt value: ", ipopt_value)

# plot the mu_optimized that are not nan
plt.plot(mu_optimized[~np.isnan(mu_optimized)], np.log10(sigma_vals[~np.isnan(mu_optimized)]), 
         'go', label='Optimized mu for fixed sigma')


plt.contourf(mu_vals, np.log10(sigma_vals), R_vals.T, levels=50)
plt.xlabel('mu')
plt.ylabel('log(sigma)')
plt.colorbar(label='R(mu, sigma^2)')

# plot optimal point
plt.plot(mu_opt, np.log10(sigma2_opt**0.5), 'ro', label='NLP point')
# set x lim
plt.xlim([-10, 10])
plt.legend()
plt.savefig("EML_contour.pdf")