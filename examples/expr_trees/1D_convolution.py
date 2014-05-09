#!/usr/bin/env python

from cvxpy import *
import numpy as np
import random

from math import pi, sqrt, exp

def gauss(n=11,sigma=1):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

np.random.seed(5)
random.seed(5)
DENSITY = 0.008
n = 1000
x = Variable(n)
# Create sparse signal.
signal = np.zeros(n)
nnz = 0
for i in range(n):
    if random.random() < DENSITY:
        signal[i] = random.uniform(0, 100)
        nnz += 1

# Gaussian kernel.
m = 51
kernel = gauss(m)

# Noisy signal.
std = 1
noise = np.random.normal(scale=std, size=n+m-1)
noisy_signal = conv(kernel, signal) + noise

gamma = Parameter(sign="positive")
fit = norm(conv(kernel, x) - noisy_signal, 2)
regularization = norm(x, 1)
constraints = [x >= 0]
gamma.value = 0.06
prob = Problem(Minimize(fit + gamma*regularization), constraints)
solver_options = {"NORMALIZE": False, "MAX_ITERS": 2500}
# result = prob.solve(solver=SCS,
#                     expr_tree=True,
#                     verbose=True,
#                     solver_specific_opts=solver_options)
# Get problem matrix.
data, dims = prob.get_problem_data(solver=SCS)

# Plot result and fit.
import matplotlib.pyplot as plt
plt.plot(range(1000), signal, label="true signal")
plt.plot(range(1000), np.asarray(noisy_signal.value[:1000, 0]), label="noisy convolution")
plt.plot(range(1000), np.asarray(x.value[:,0]), label="recovered signal")
plt.legend(loc='upper right')
plt.show()
