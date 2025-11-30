import pdb

import numpy as np
from util import branch_matrices, create_admittance_matrices, plot_power_flows

import cvxpy as cp

# -----------------------------------------------------------------------------------
#                             Define prob data
# -----------------------------------------------------------------------------------
N = 9
p_min = np.zeros(N)
p_max = np.zeros(N)
q_min = np.zeros(N)
q_max = np.zeros(N)
p_min[[0, 1, 2]] = [10, 10, 10]
p_max[[0, 1, 2]] = [250, 300, 270]
q_min[[0, 1, 2]] = [-5, -5, -5]
p_min[[4,6,8]] = p_max[[4,6,8]] = [-54, -60, -75]
q_min[[4,6,8]] = q_max[[4,6,8]] = [-18, -21, -30]
v_min, v_max = 0.9, 1.1
G1_series, B1_series, G1_sh, B1_sh = branch_matrices()

# these have different signs etc
G0_series, B0_series, G0_sh, B0_sh = create_admittance_matrices()
#G = G0 + G_sh
#B = B0 + B_sh

D0 = G0_series + G1_series
d0 = np.diag(D0).reshape((N, 1))
#import pdb 
#pdb.set_trace()

g_sh = np.diag(G1_sh).reshape((N, 1))
b_sh = np.diag(B1_sh).reshape((N, 1))

# -----------------------------------------------------------------------------------
#                         Define optimization prob
# -----------------------------------------------------------------------------------
theta, P, Q = cp.Variable((N, 1)), cp.Variable((N, N)), cp.Variable((N, N))
v = cp.Variable((N, 1), bounds=[v_min, v_max])
p = cp.Variable(N, bounds=[p_min, p_max])
q = cp.Variable(N, bounds=[q_min, q_max])
C, S = cp.cos(theta - theta.T), cp.sin(theta - theta.T) 

#import pdb 
#pdb.set_trace()

constr = [theta[0] == 0,           #   should figure out why it crashes then
          p == cp.sum(P, axis=1), 
          #+ #  if we explicitly incorporate this ipopt crashes     cp.multiply(v ** 2, g_sh).T, 
          q == cp.sum(Q, axis=1) - cp.multiply(v ** 2, b_sh).T,
          P == cp.multiply(v ** 2, G1_series) -
               cp.multiply(v @ v.T, cp.multiply(G1_series, C) + cp.multiply(B1_series, S)),
          Q == -cp.multiply(v ** 2, B1_series) -
               cp.multiply(v @ v.T, cp.multiply(G1_series, S) - cp.multiply(B1_series, C))]
cost = (0.11 * p[0]**2 + 5 * p[0] + 150 + 0.085 * p[1]**2 + 1.2 * p[1] + 600 +
        0.1225 * p[2]**2 + p[2] + 335)
prob = cp.Problem(cp.Minimize(cost), constr)

# -----------------------------------------------------------------------------------
#              Solve prob (initialize to 1.0 p.u. and 0 degrees)
# -----------------------------------------------------------------------------------
v.value = np.ones((N, 1))  # TODO: look up why this matters. v is auto-initialized to 1, right? 
                           # Perhaps the order changes of value propagation changes?
theta.value = np.zeros((N, 1)) # Why does this matter? Don't we auto-initialize to zero?


result = prob.solve(solver=cp.IPOPT, verbose=True, nlp=True, derivative_test='second-order', 
                    least_square_init_duals='no')


print(f"\nSolver status: {prob.status}")
print(f"Optimal objective value: {prob.value:.2f}")
print("\nGeneration (MW):")
for i in range(3):
    print(f"  Bus {i+1}: P={p.value[i]:.2f}, Q={q.value[i]:.2f}")



print("\nVoltage magnitudes (p.u.):")
for i in range(N):
    print(f"  Bus {i+1}: {v.value[i, 0]:.4f}")
#
print("\nVoltage angles (degrees):")
for i in range(N):
    print(f"  Bus {i+1}: {np.rad2deg(theta.value[i, 0]):.2f}")

P_flow = P.value
pdb.set_trace()
plot_power_flows(P_flow)