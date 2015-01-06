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
n = 100000
NUM_SPIKES = 6.0
DENSITY = NUM_SPIKES/n
x = Variable(n)
# Create sparse signal.
HEIGHT = n/10
true_x = np.zeros((n,1))
nnz = 0
for i in range(n):
    if random.random() < DENSITY and NUM_SPIKES > nnz:
        true_x[i] = random.uniform(0, HEIGHT)
        nnz += 1

# Gaussian kernel.
m = n+1
kernel = gauss(m, m/10, 1, 1e-6)
#kernel = np.sinc(np.linspace(-m/100, m/100, m))

# Noisy signal.
SNR = 20
signal = conv(kernel, true_x)
sigma = norm(signal,2).value/(SNR*sqrt(n+m-1))
noise = np.random.normal(scale=sigma, size=n+m-1)
print("SNR", norm(signal,2).value/norm(noise,2).value)
noisy_signal = signal + noise

gamma = Parameter(sign="positive")
gamma.value = 0
fit = sum_squares(conv(kernel, x) - noisy_signal)
constraints = [x >= 0]
prob = Problem(Minimize(fit),
               constraints)
# result = prob.solve(solver=ECOS, verbose=True)
# print("solve time", prob.solve_time)
# result = prob.solve(solver=SCS,
#                     verbose=True,
#                     max_iters=2500,
#                     eps=1e-3,
#                     use_indirect=False)
# print("solve time", prob.solve_time)
# print("true signal fit", fit.value)
result = prob.solve(solver=SCS_MAT_FREE,
                    verbose=True,
                    max_iters=2500,
                    equil_steps=10,
                    eps=1e-3,
                    cg_rate=2)
print("solve time", prob.solve_time)
print("recovered x fit", fit.value)

print("nnz =", np.sum(x.value >= 1))
print("max =", np.max(np.max(x.value)))
# Plot result and fit.
# Assumes convolution kernel is centered around m/2.
t = range(n+m-1)
import matplotlib.pyplot as plt
plt.figure()
ax1 = plt.subplot(2, 1, 1)
true_x_padded = np.vstack([np.zeros((m/2,1)), true_x, np.zeros((m/2,1))])
x_padded = np.vstack([np.zeros((m/2,1)), x.value[:,0], np.zeros((m/2,1))])
lns1 = ax1.plot(t, true_x_padded, label="true x")
ax1.set_ylabel('true x')
ax2 = ax1.twinx()
lns2 = ax2.plot(t, x_padded, label="recovered x", color="red")
ax2.set_ylabel('recovered x')
ax2.set_ylim([0, np.max(x_padded)])

# added these three lines
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper right')

plt.subplot(2, 1, 2)
plt.plot(t, np.asarray(noisy_signal.value[:, 0]), label="signal")
plt.legend(loc='upper right')
plt.show()
