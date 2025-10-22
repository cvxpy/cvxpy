import numpy as np
from scipy.sparse import csr_matrix, diags

import cvxpy as cp

# Number of buses
N = 9
# Real generation bounds
P_Gen_lb = np.zeros(N)
P_Gen_lb[[0, 1, 2]] = [10, 10, 10]
P_Gen_ub = np.zeros(N)
P_Gen_ub[[0, 1, 2]] = [250, 300, 270]
# Reactive generation bounds
Q_Gen_lb = np.zeros(N)
Q_Gen_lb[[0, 1, 2]] = [-5, -5, -5]
Q_Gen_ub = np.zeros(N)
Q_Gen_ub[[0, 1, 2]] = [300, 300, 300]
# Power demand (real and reactive)
P_Demand = np.zeros(N)
P_Demand[[4, 6, 8]] = [54, 60, 75]
Q_Demand = np.zeros(N)
Q_Demand[[4, 6, 8]] = [18, 21, 30]

# Branch data: (from_bus, to_bus, resistance, reactance, susceptance)
# Note: Julia uses 1-indexing, Python uses 0-indexing
branch_data = np.array([
    [0, 3, 0.0, 0.0576, 0.0],
    [3, 4, 0.017, 0.092, 0.158],
    [5, 4, 0.039, 0.17, 0.358],
    [2, 5, 0.0, 0.0586, 0.0],
    [5, 6, 0.0119, 0.1008, 0.209],
    [7, 6, 0.0085, 0.072, 0.149],
    [1, 7, 0.0, 0.0625, 0.0],
    [7, 8, 0.032, 0.161, 0.306],
    [3, 8, 0.01, 0.085, 0.176],
])

M = branch_data.shape[0]  # Number of branches
base_MVA = 100

# Build incidence matrix A
from_bus = branch_data[:, 0].astype(int)
to_bus = branch_data[:, 1].astype(int)
A = csr_matrix((np.ones(M), (from_bus, np.arange(M))), shape=(N, M)) + \
    csr_matrix((-np.ones(M), (to_bus, np.arange(M))), shape=(N, M))

# Network impedance
z = (branch_data[:, 2] + 1j * branch_data[:, 3]) / base_MVA

# Bus admittance matrix Y_0
Y_0 = A @ diags(1.0 / z) @ A.T

# Shunt admittance from line charging
y_sh = 0.5 * (1j * branch_data[:, 4]) * base_MVA
Y_sh_diag = np.array((A @ diags(y_sh) @ A.T).diagonal()).flatten()
Y_sh = diags(Y_sh_diag)

# Full bus admittance matrix
Y = Y_0 + Y_sh
Y_dense = Y.toarray()

# Decision variables
# Voltage: magnitude and angle for each bus
V_mag = cp.Variable(N, bounds=[0.9, 1.1])
V_ang = cp.Variable(N)

# Power generation: real and reactive
P_G = cp.Variable(N, bounds=[P_Gen_lb, P_Gen_ub])
Q_G = cp.Variable(N, bounds=[Q_Gen_lb, Q_Gen_ub])
# Constraints list
constraints = []
# Reference bus (bus 1, index 0): fix angle to 0
constraints.append(V_ang[0] == 0)

# Power flow equations: S_G - S_Demand = V * conj(Y * V)
# Breaking into real and imaginary parts:
# P_G - P_Demand = Real(V * conj(Y * V))
# Q_G - Q_Demand = Imag(V * conj(Y * V))

# For each bus i:
# P_i = V_mag[i] * sum_j(V_mag[j] * (G[i,j]*cos(theta_i - theta_j) + B[i,j]*sin(theta_i - theta_j)))
# Q_i = V_mag[i] * sum_j(V_mag[j] * (G[i,j]*sin(theta_i - theta_j) - B[i,j]*cos(theta_i - theta_j)))

G = np.real(Y_dense)  # Conductance matrix
B = np.imag(Y_dense)  # Susceptance matrix

# Compute voltage phasors in matrix form
# V_mag is shape (N,), V_ang is shape (N,)
# Create matrices for all pairwise angle differences
theta_diff = V_ang[:, None] - V_ang[None, :]  # Shape (N, N)

# Vectorized power injection calculations
# Real power injections for all buses
cos_theta = cp.cos(theta_diff)
sin_theta = cp.sin(theta_diff)

P_injection = cp.multiply(
    V_mag,
    cp.sum(cp.multiply(V_mag[None, :], G @ cos_theta + B @ sin_theta), axis=1)
)

# Reactive power injections for all buses  
Q_injection = cp.multiply(
    V_mag,
    cp.sum(cp.multiply(V_mag[None, :], G @ sin_theta - B @ cos_theta), axis=1)
)

# Add vectorized constraints
constraints.append(P_G - P_Demand == P_injection)
constraints.append(Q_G - Q_Demand == Q_injection)

# Objective: minimize quadratic generation cost
objective = cp.Minimize(0)

# Create and solve problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True,
              derivative_test='second-order')
              #fixed_variable_treatment='')

print(f"\nOptimal objective value: {problem.value:.2f}")
print("\nGeneration (MW):")
print(f"  Bus 1: P={P_G.value[0]:.2f}, Q={Q_G.value[0]:.2f}")
print(f"  Bus 2: P={P_G.value[1]:.2f}, Q={Q_G.value[1]:.2f}")
print(f"  Bus 3: P={P_G.value[2]:.2f}, Q={Q_G.value[2]:.2f}")
print(f"\nVoltage magnitudes: {V_mag.value}")
print(f"Voltage angles (deg): {np.rad2deg(V_ang.value)}")
