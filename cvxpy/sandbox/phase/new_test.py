

import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp

# Set up random number generator for reproducibility
rng = np.random.default_rng(42)

n = 64
m = int(3 * n)

# data
x0 = rng.random(n) + 1j * rng.random(n)
A = rng.random((m, n)) + 1j * rng.random((m, n))
x0_real, x0_imag = np.real(x0), np.imag(x0)
A_real, A_imag = np.real(A), np.imag(A)
y = np.abs(A @ x0)
B = np.hstack([A_real, -A_imag])
C = np.hstack([A_imag, A_real])

x_tilde = cp.Variable(2 * n)
#cost = cp.norm2(cp.square(B @ x_tilde) + cp.square(C @ x_tilde) - cp.square(y))
cost = cp.norm1((B @ x_tilde) ** 2 + (C @ x_tilde) ** 2 - y ** 2)
#cost = cp.sum(cp.square(cp.square(B @ x_tilde) + cp.square(C @ x_tilde) - cp.square(y)))

# solve the problem
prob = cp.Problem(cp.Minimize(cost))
result = prob.solve(solver=cp.IPOPT, verbose=True, nlp=True, derivative_test='none', 
                    least_square_init_duals='no')



# check if recoverd signal
x_recovered = x_tilde[0:n] + 1j * x_tilde[n:2*n]
y_recovered = np.abs(A @ x_recovered.value)

x = np.vstack([x_tilde.value[:n].T, x_tilde.value[n:].T])
fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(10, 8), dpi=150)
tan = np.array(x[1, :] / x[0, :])
angle = np.arctan(tan)
tan0 = x0_imag / x0_real
angle0 = np.arctan(tan0)
ax0.plot(angle0)
ax0.plot(angle, "r")
ax1.plot(np.array(np.power(x0_real, 2) + np.power(x0_imag, 2)))
ax1.plot(np.array(np.power(x[0, :], 2) + np.power(x[1, :], 2)), "r--")
ax0.set_ylabel("Phase", fontsize=14)
ax1.set_ylabel("Amplitude", fontsize=14)

# add legend above first figure
ax0.legend(["Original signal", "Recovered signal"], loc='upper center', bbox_to_anchor=(0.5, 1.15),
            ncol=2, fontsize=15)

plt.tight_layout()
plt.savefig("phase_retrieval.pdf")