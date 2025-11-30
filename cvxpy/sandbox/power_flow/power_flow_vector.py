import numpy as np
from util import create_admittance_matrices

import cvxpy as cp

# -----------------------------------------------------------------------------------
#                             Define problem data
# -----------------------------------------------------------------------------------
N = 9
p_min = np.zeros(N)
p_min[[0, 1, 2]] = [10, 10, 10]         
p_max = np.zeros(N)
p_max[[0, 1, 2]] = [250, 300, 270]
q_min = np.zeros(N)
q_min[[0, 1, 2]] = [-5, -5, -5]
q_max = np.zeros(N)
q_max[[0, 1, 2]] = [300, 300, 300]
p_ext = np.zeros(N)
p_ext[[4, 6, 8]] = [54, 60, 75]
q_ext = np.zeros(N)
q_ext[[4, 6, 8]] = [18, 21, 30]
G, B = create_admittance_matrices()

# -----------------------------------------------------------------------------------
#                         Define optimization problem
# -----------------------------------------------------------------------------------
theta = cp.Variable((N, 1))
v = cp.Variable(N, bounds=[0.9, 1.1])
p = cp.Variable(N, bounds=[p_min, p_max])
q = cp.Variable(N, bounds=[q_min, q_max])
C, S = cp.cos(theta - theta.T), cp.sin(theta - theta.T)   
constraints = [theta[0] == 0,
               p - p_ext == cp.multiply(v, (cp.multiply(G, C) + cp.multiply(B, S)) @ v),
               q - q_ext == cp.multiply(v, (cp.multiply(G, S) - cp.multiply(B, C)) @ v)
               ]
cost = (0.11 * p[0]**2 + 5 * p[0] + 150 +
        0.085 * p[1]**2 + 1.2 * p[1] + 600 +
        0.1225 * p[2]**2 + p[2] + 335)
problem = cp.Problem(cp.Minimize(cost), constraints)

# -----------------------------------------------------------------------------------
#              Solve problem (initialize to 1.0 p.u. and 0 degrees)
# -----------------------------------------------------------------------------------
v.value = np.ones(N)  
theta.value = np.zeros((N, 1)) 
result = problem.solve(solver=cp.IPOPT, verbose=True, nlp=True,
                       derivative_test='none')
                      
print(f"\nSolver status: {problem.status}")
print(f"Optimal objective value: {problem.value:.2f}")
print("\nGeneration (MW):")
for i in range(3):
    print(f"  Bus {i+1}: P={p.value[i]:.2f}, Q={q.value[i]:.2f}")

print("\nVoltage magnitudes (p.u.):")
for i in range(N):
    print(f"  Bus {i+1}: {v.value[i]:.4f}")

print("\nVoltage angles (degrees):")
for i in range(N):
    print(f"  Bus {i+1}: {np.rad2deg(theta.value[i, 0]):.2f}")
