#!/usr/bin/env python

from cvxpy import *
import numpy as np
import random
from math import pi, sqrt, exp

def gauss(n=11,sigma=1, scale=1, min_val=1):
    r = range(-int(n/2),int(n/2)+1)
    return [max(scale /(sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)), min_val) for x in r]

np.random.seed(5)
random.seed(5)
n = 1000
DENSITY = 6.0/n
x = Variable(n)
# Create sparse signal.
HEIGHT = n/10
true_x = np.zeros((n,1))
nnz = 0
for i in range(n):
    if random.random() < DENSITY:
        true_x[i] = random.uniform(0, HEIGHT)
        nnz += 1

# Gaussian kernel.
m = n+1
kernel = gauss(m, m/10, 100, 0.001)
#kernel = np.sinc(np.linspace(-m/100, m/100, m))

# Noisy signal.
SNR = 20
signal = conv(kernel, true_x)
sigma = norm(signal,2).value/(SNR*sqrt(n+m-1))
noise = np.random.normal(scale=sigma, size=n+m-1)
print("SNR", norm(signal,2).value/norm(noise,2).value)
noisy_signal = signal + noise

gamma = Parameter(sign="positive")
fit = norm(conv(kernel, x) - noisy_signal, 2)
constraints = [x >= 0]
prob = Problem(Minimize(fit),
               constraints)
result = prob.solve(solver=ECOS, verbose=True)
# result = prob.solve(solver=SCS,
#                     verbose=True,
#                     max_iters=2500,
#                     eps=1e-3,
#                     use_indirect=False)
# print("true signal fit", fit.value)
# result = prob.solve(solver=SCS_MAT_FREE,
#                     verbose=True,
#                     max_iters=2000,
#                     equil_steps=1,
#                     eps=1e-4,
#                     cg_rate=2)
print("recovered x fit", fit.value)

# Plot result and fit.
# Assumes convolution kernel is centered around m/2.
t = range(n+m-1)
import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2, 1, 1)
true_x_padded = np.vstack([np.zeros((m/2,1)), true_x, np.zeros((m/2,1))])
x_padded = np.vstack([np.zeros((m/2,1)), x.value[:,0], np.zeros((m/2,1))])
plt.plot(t, true_x_padded, label="true x")
plt.plot(t, x_padded, label="recovered x", color="red")
plt.legend(loc='upper right')
plt.subplot(2, 1, 2)
plt.plot(t, np.asarray(noisy_signal.value[:, 0]), label="signal")
plt.legend(loc='upper right')
plt.show()
