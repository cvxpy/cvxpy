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

# Extract conductance and susceptance matrices
G = np.real(Y_dense)  # Conductance matrix
B = np.imag(Y_dense)  # Susceptance matrix

# Decision variables with bounds
# Voltage magnitude and angle for each bus
V_mag = cp.Variable(N, bounds=[0.9, 1.1])
V_ang = cp.Variable(N)

# Power generation: real and reactive
P_G = cp.Variable(N, bounds=[P_Gen_lb, P_Gen_ub])
Q_G = cp.Variable(N, bounds=[Q_Gen_lb, Q_Gen_ub])

# Initialize variables (important for nonlinear problems)
V_mag.value = np.ones(N)  # Start at 1.0 p.u.
V_ang.value = np.zeros(N)  # Start at 0 degrees
P_G.value = (P_Gen_lb + P_Gen_ub) / 2  # Start at midpoint
Q_G.value = (Q_Gen_lb + Q_Gen_ub) / 2

# Constraints list
constraints = []
#constraints.append(P_G >= P_Gen_lb)
#constraints.append(P_G <= P_Gen_ub)
#constraints.append(Q_G >= Q_Gen_lb)
#constraints.append(Q_G <= Q_Gen_ub)
# Reference bus (bus 1, index 0): fix angle to 0
constraints.append(V_ang[0] == 0)

# Power flow equations - fully vectorized
# Create angle difference matrix: theta_diff[i,j] = theta_i - theta_j
# Using cp.reshape with explicit order='F' for column-major
V_ang_col = cp.reshape(V_ang, (N, 1), order='F')  # Shape (N, 1)
V_ang_row = cp.reshape(V_ang, (1, N), order='F')  # Shape (1, N)
theta_diff = V_ang_col - V_ang_row  # Shape (N, N)

# Compute cos and sin of all angle differences
cos_theta = cp.cos(theta_diff)  # Shape (N, N)
sin_theta = cp.sin(theta_diff)  # Shape (N, N)

# Compute the matrix products
# G * cos(theta_diff) + B * sin(theta_diff) gives the real part coefficients
# G * sin(theta_diff) - B * cos(theta_diff) gives the reactive part coefficients
real_coeffs = cp.multiply(G, cos_theta) + cp.multiply(B, sin_theta)  # Element-wise
reactive_coeffs = cp.multiply(G, sin_theta) - cp.multiply(B, cos_theta)

# For each bus i, compute: sum_j(V_mag[j] * coeffs[i,j])
# This is matrix-vector multiplication: coeffs @ V_mag
# Compute: V_mag[i] * sum_j(V_mag[j] * coeffs[i,j])
# = V_mag[i] * (coeffs[i,:] @ V_mag)
P_injection = cp.multiply(V_mag, real_coeffs @ V_mag)  # Shape (N,)
Q_injection = cp.multiply(V_mag, reactive_coeffs @ V_mag)  # Shape (N,)

# Power balance: Generation - Demand = Injection (vectorized for all buses)
constraints.append(P_G - P_Demand == P_injection)
constraints.append(Q_G - Q_Demand == Q_injection)

# Objective: minimize quadratic generation cost
objective = cp.Minimize(
    0.11 * P_G[0]**2 + 5 * P_G[0] + 150 +
    0.085 * P_G[1]**2 + 1.2 * P_G[1] + 600 +
    0.1225 * P_G[2]**2 + P_G[2] + 335
)

# Create and solve problem
problem = cp.Problem(objective, constraints)

# Solve with IPOPT
result = problem.solve(solver=cp.IPOPT, verbose=True, nlp=True,
                       derivative_test='first-order',)
                       #fixed_variable_treatment='relax_bounds')

print(f"\nSolver status: {problem.status}")
print(f"Optimal objective value: {problem.value:.2f}")
print("\nGeneration (MW):")
for i in range(3):
    print(f"  Bus {i+1}: P={P_G.value[i]:.2f}, Q={Q_G.value[i]:.2f}")

print("\nVoltage magnitudes (p.u.):")
for i in range(N):
    print(f"  Bus {i+1}: {V_mag.value[i]:.4f}")

print("\nVoltage angles (degrees):")
for i in range(N):
    print(f"  Bus {i+1}: {np.rad2deg(V_ang.value[i]):.2f}")
