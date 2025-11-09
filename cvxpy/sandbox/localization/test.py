import numpy as np

import cvxpy as cp

np.random.seed(42)

# Problem setup
m = 10                
dim = 2               
x_true = np.array([2.0, -1.5])  

# Generate random anchor positions
a = np.random.uniform(-5, 5, (m, dim))

# Generate noise and corresponding range measurements
noise_std = 0  # standard deviation of noise
v = np.random.normal(0, noise_std, m)
rho = np.linalg.norm(a - x_true, axis=1) + v

x = cp.Variable(2, name='x')
t = cp.Variable(m, name='t')


# express vectorized (this yields a solve time that is 100 times faster than the loop
# below)
constraints = [t == cp.sqrt(cp.sum(cp.square(x - a), axis=1))]

# slow approach 
#constraints = []
#for i in range(m):
#   constraints.append(t[i] - cp.sqrt(cp.sum_squares(x - a[i, :])) == 0)

objective = cp.Minimize(cp.sum_squares(t - rho))
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True, derivative_test='none',
              least_square_init_duals='no')


print("\nEstimated position x_est:\n", x.value)
print("True position x_true:\n", x_true)