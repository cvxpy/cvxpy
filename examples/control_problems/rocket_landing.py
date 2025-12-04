import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp

# Parameters (all normalized to be dimensionless)
h_0 = 1                      # Initial height
v_0 = 0                      # Initial velocity
m_0 = 1.0                    # Initial mass
m_T = 0.6                    # Final mass
g_0 = 1                      # Gravity at the surface
h_c = 500                    # Used for drag
c = 0.5 * np.sqrt(g_0 * h_0)  # Thrust-to-fuel mass
D_c = 0.5 * 620 * m_0 / g_0  # Drag scaling
u_t_max = 3.5 * g_0 * m_0    # Maximum thrust
T_max = 0.2                  # Number of seconds
T = 250                  # Number of time steps
dt = 0.2 / T                 # Time per discretized step

# Create variables with bounds
x_v = cp.Variable(T, nonneg=True, name="velocity")  # Velocity >= 0
x_h = cp.Variable(T, nonneg=True, name="height")  # Height >= 0
x_m = cp.Variable(T, bounds=[m_T, None], name="mass") # Mass (lower bound added as constraint)
u_t = cp.Variable(T, bounds=[0, u_t_max], name="thrust")  # Thrust (bounds added as constraints)

# Set initial values (warm start) for the variables
# In CVXPY, we set these values before solving
x_v.value = np.full(T, v_0)        # Start with initial velocity
x_h.value = np.full(T, h_0)        # Start with initial height
x_m.value = np.full(T, m_0)        # Start with initial mass
u_t.value = np.zeros(T)            # Start with zero thrust

# Objective: Maximize altitude at end of time of flight
objective = cp.Maximize(x_h[T-1])

# Constraints list
constraints = []

# Boundary conditions
constraints += [
    x_v[0] == v_0,
    x_h[0] == h_0,
    x_m[0] == m_0,
    u_t[T-1] == 0.0
]

# Variable bounds
#constraints += [x_m >= m_T]                    # Mass lower bound
#constraints += [u_t >= 0, u_t <= u_t_max]      # Thrust bounds

# Dynamical equations using vectorized constraints
# Note: We use indices [1:T] for current time and [0:T-1] for previous time

# Height dynamics: dh/dt = v
# (x_h[t] - x_h[t-1]) / dt = x_v[t-1] for t in 2:T
constraints += [
    (x_h[1:T] - x_h[0:T-1]) / dt == x_v[0:T-1]
]

# Velocity dynamics: dv/dt = (u_t - D) / x_m - g
# where D(x_h, x_v) = D_c * x_v^2 * exp(-h_c * (x_h - h_0) / h_0)
# and g(x_h) = g_0 * (h_0 / x_h)^2
# 
# For vectorization, we compute drag and gravity terms
# Note: cp.exp might not be available in all CVXPY versions, 
# so this might need to be reformulated for some solvers

# Drag force vectorized
D_vec = D_c * cp.multiply(
    cp.multiply(x_v[0:T-1], x_v[0:T-1]), 
    cp.exp(-h_c * (x_h[0:T-1] - h_0) / h_0)
)

# Gravity force vectorized
g_vec = g_0 * cp.power(h_0 / x_h[0:T-1], 2)

# Velocity dynamics constraint
constraints += [
    (x_v[1:T] - x_v[0:T-1]) / dt == (u_t[0:T-1] - D_vec) / x_m[0:T-1] - g_vec
]

# Mass dynamics: dm/dt = -u_t / c
constraints += [
    (x_m[1:T] - x_m[0:T-1]) / dt == -u_t[0:T-1] / c
]

# Create and solve the problem
problem = cp.Problem(objective, constraints)

# Solve using IPOPT or another nonlinear solver with warm start enabled
# Note: You need to have IPOPT installed and accessible through CVXPY
# The warm_start parameter tells the solver to use the initial values we set

result = problem.solve(solver=cp.IPOPT, verbose=True, nlp=True, derivative_test='none')
solver_used = "IPOPT"

print(f"Solver used: {solver_used}")
print(f"Status: {problem.status}")
print(f"Optimal value (max altitude): {problem.value:.5f}")

# Extract solution values
x_h_sol = x_h.value
x_v_sol = x_v.value
x_m_sol = x_m.value
u_t_sol = u_t.value

# Plot the trajectory
time_points = np.arange(T) * dt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(time_points, x_h_sol)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Altitude')
axes[0, 0].grid(True)

axes[0, 1].plot(time_points, x_m_sol)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Mass')
axes[0, 1].grid(True)

axes[1, 0].plot(time_points, x_v_sol)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Velocity')
axes[1, 0].grid(True)

axes[1, 1].plot(time_points, u_t_sol)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Thrust')
axes[1, 1].grid(True)

plt.suptitle('Rocket Optimal Control Trajectory')
plt.tight_layout()
plt.show()

# Print some statistics
print(f"\nFinal altitude: {x_h_sol[-1]:.5f}")
print(f"Final velocity: {x_v_sol[-1]:.5f}")
print(f"Final mass: {x_m_sol[-1]:.5f}")
print(f"Max thrust used: {np.max(u_t_sol):.5f}")
print(f"Total fuel consumed: {(m_0 - x_m_sol[-1]):.5f}")
