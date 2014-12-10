#!/usr/bin/env python

from cvxpy import *
import numpy as np
import random

from math import pi, sqrt, exp

def gauss(n=11,sigma=1, scale=1):
    r = range(-int(n/2),int(n/2)+1)
    return [scale / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

np.random.seed(5)
random.seed(5)
n = 100000
DENSITY = 6.0/n
x = Variable(n)
# Create sparse signal.
HEIGHT = 100
signal = np.zeros(n)
nnz = 0
for i in range(n):
    if random.random() < DENSITY:
        signal[i] = random.uniform(0, HEIGHT)
        nnz += 1

# Gaussian kernel.
m = 100001
kernel = gauss(m, m/10, 1)

# Noisy signal.
std = 1
noise = np.random.normal(scale=std, size=n+m-1)
noisy_signal = conv(kernel, signal) + noise

gamma = Parameter(sign="positive")
fit = sum_squares(conv(kernel, x) - noisy_signal)
reg = norm(x, 1)
constraints = [x >= 0]
gamma.value = 0.5
prob = Problem(Minimize(fit),
               constraints)
# result = prob.solve(solver=ECOS, verbose=True)
# print "true signal fit", fit.value
result = prob.solve(solver=SCS_MAT_FREE,
                    verbose=True,
                    max_iters=500,
                    equil_steps=1,
                    eps=1e-3)
print "recovered signal fit", fit.value

# # Plot result and fit.
# import matplotlib.pyplot as plt
# plt.plot(range(n), signal, label="true signal")
# plt.plot(range(n), np.asarray(noisy_signal.value[:n, 0]), label="noisy convolution")
# plt.plot(range(n), np.asarray(x.value[:,0]), label="recovered signal")
# plt.legend(loc='upper right')
# plt.show()
