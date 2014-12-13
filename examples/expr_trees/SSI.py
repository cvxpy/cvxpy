#!/usr/bin/env python

from cvxpy import *
import numpy as np
import scipy.sparse as sp
import random
from math import pi, sqrt, exp

def gauss(n=11,sigma=1, scale=1):
    r = range(-int(n/2),int(n/2)+1)
    return [scale / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]
    #return [scale*exp(-float(x)**2/(2*sigma**2)) for x in r]

np.random.seed(5)
random.seed(5)
n = 1000
# DENSITY = 6.0/n
# x = Variable(n)
# # Create sparse signal.
# HEIGHT = n/10
# true_x = np.zeros((n,1))
# nnz = 0
# for i in range(n):
#     # if random.random() < DENSITY:
#     #     true_x[i] = random.uniform(0, HEIGHT)
#     #     nnz += 1

k = 10
true_x = np.zeros((k,1))
for i in range(k):
    true_x[i] = random.choice([-1.0,1.0])

frac = n/k
A = sp.kron(sp.eye(k), np.ones((frac,1)))

# Gaussian kernel.
m = n+1
kernel = gauss(m, m/10.0, 1)

# Noisy signal.
SNR = 0.5
signal = conv(kernel, A.dot(true_x))
sigma = norm(signal,2).value/(SNR*sqrt(n+m-1))
noise = np.random.normal(scale=sigma, size=n+m-1)
print("SNR", norm(signal,2).value/norm(noise,2).value)
noisy_signal = signal + noise

x = Variable(n)
gamma = Parameter(sign="positive")
fit = norm(conv(kernel, x) - noisy_signal, 2)
reg = 0.1*tv(x)
constraints = [-1 <= x, x <= 1]
prob = Problem(Minimize(fit),
               constraints)
# result = prob.solve(solver=ECOS, verbose=True)
# # result = prob.solve(solver=SCS,
# #                     verbose=True,
# #                     max_iters=2500,
# #                     eps=1e-3,
# #                     use_indirect=True)
# print("true signal fit", fit.value)
result = prob.solve(solver=SCS_MAT_FREE,
                    verbose=True,
                    max_iters=1000,
                    equil_steps=10,
                    eps=1e-4,
                    cg_rate=2)
# print("recovered x fit", fit.value)

# Timings:
# n=m=1e5, 161 sec for 500*1.51 CG steps.
# At 10 GFlops should be .160 seconds.
# n=m=1e5, eps=1e-4, 155 sec for 580*1.02 CG steps.

# Plot result and fit.
# Assumes convolution kernel is centered around m/2.
t = range(n+m-1)
import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2, 1, 1)
true_x_padded = np.vstack([np.zeros((m/2,1)), A.dot(true_x), np.zeros((m/2,1))])
x_padded = np.vstack([np.zeros((m/2,1)), x.value[:,0], np.zeros((m/2,1))])
plt.plot(t, true_x_padded, label="true x")
plt.plot(t, x_padded, label="recovered x", color="red")
plt.legend(loc='upper right')
plt.subplot(2, 1, 2)
plt.plot(t, np.asarray(noisy_signal.value[:, 0]), label="signal")
plt.legend(loc='upper right')
plt.show()
