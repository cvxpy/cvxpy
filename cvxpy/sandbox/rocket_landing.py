import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp

# Data - all parameters normalized to be dimensionless
h_0 = 1                      # Initial height
v_0 = 0                      # Initial velocity
m_0 = 1.0                    # Initial mass
m_T = 0.6                    # Final mass
g_0 = 1                      # Gravity at the surface
h_c = 500                    # Used for drag
c = 0.5 * np.sqrt(g_0 * h_0) # Thrust-to-fuel mass
D_c = 0.5 * 620 * m_0 / g_0  # Drag scaling
u_t_max = 3.5 * g_0 * m_0    # Maximum thrust
T_max = 0.2                  # Number of seconds
T = 2                        # Number of time steps
dt = 0.2 / T                 # Time per discretized step

# Create variables
x_v = cp.Variable(T, bounds=[0, np.inf])  # Velocity
x_h = cp.Variable(T, bounds=[0, np.inf])  # Height  
x_m = cp.Variable(T)                      # Mass
u_t = cp.Variable(T, bounds=[0, u_t_max]) # Thrust

# Set starting values (equivalent to JuMP's start parameter)
x_v.value = np.full(T, v_0)        # start = v_0
x_h.value = np.full(T, h_0)        # start = h_0
x_m.value = np.full(T, m_0)        # start = m_0
u_t.value = np.zeros(T)            # start = 0

# Initialize constraints list
constraints = []

# Boundary conditions
constraints.append(x_v[0] == v_0)
constraints.append(x_h[0] == h_0)
constraints.append(x_m[0] == m_0)
constraints.append(u_t[T-1] == 0.0)

# Mass constraints
constraints.append(x_m >= m_T)

# Thrust constraints
constraints.append(u_t <= u_t_max)

# Dynamical equations as constraints
for t in range(1, T):
    # Rate of ascent: dx_h/dt = x_v
    # Using backward difference: (x_h[t] - x_h[t-1])/dt = x_v[t-1]
    constraints.append((x_h[t] - x_h[t-1]) / dt == x_v[t-1])
    
    # Acceleration: dx_v/dt = (u_t - D(x_h, x_v))/x_m - g(x_h)
    # Where D(x_h, x_v) = D_c * x_v^2 * exp(-h_c * (x_h - h_0) / h_0)
    # And g(x_h) = g_0 * (h_0 / x_h)^2
    # Using backward difference: (x_v[t] - x_v[t-1])/dt = ...
    drag_force = D_c * cp.square(x_v[t-1]) * cp.exp(-h_c * (x_h[t-1] - h_0) / h_0)
    gravity_force = g_0 * cp.square(h_0 / x_h[t-1])
    
    constraints.append(
        (x_v[t] - x_v[t-1]) / dt == (u_t[t-1] - drag_force) / x_m[t-1] - gravity_force
    )
    
    # Rate of mass loss: dx_m/dt = -u_t/c
    # Using backward difference: (x_m[t] - x_m[t-1])/dt = -u_t[t-1]/c
    constraints.append((x_m[t] - x_m[t-1]) / dt == -u_t[t-1] / c)

# Objective: maximize altitude at end of time of flight
objective = cp.Maximize(x_h[T-1])

# Create and solve problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)

# Check if solution was found
if problem.status == cp.OPTIMAL:
    print(f"Optimal value: {problem.value}")
    print(f"Final altitude: {x_h.value[T-1]}")
    print(f"Final mass: {x_m.value[T-1]}")
    print(f"Final velocity: {x_v.value[T-1]}")
else:
    print(f"Problem status: {problem.status}")

# Plot results if solution found
if problem.status == cp.OPTIMAL:
    # Create time array
    time = np.arange(T) * dt
    
    # Create figure with subplots
    plt.figure(figsize=(12, 8))
    
    # Plot altitude
    plt.subplot(2, 2, 1)
    plt.plot(time, x_h.value)
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude')
    plt.grid(True)
    
    # Plot mass
    plt.subplot(2, 2, 2)
    plt.plot(time, x_m.value)
    plt.xlabel('Time (s)')
    plt.ylabel('Mass')
    plt.grid(True)
    
    # Plot velocity
    plt.subplot(2, 2, 3)
    plt.plot(time, x_v.value)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.grid(True)
    
    # Plot thrust
    plt.subplot(2, 2, 4)
    plt.plot(time, u_t.value)
    plt.xlabel('Time (s)')
    plt.ylabel('Thrust')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print("\nSolution Statistics:")
    print(f"Maximum altitude reached: {np.max(x_h.value):.6f}")
    print(f"Maximum velocity reached: {np.max(x_v.value):.6f}")
    print(f"Maximum thrust used: {np.max(u_t.value):.6f}")
    print(f"Total fuel consumed: {m_0 - x_m.value[T-1]:.6f}")
else:
    print("No optimal solution found. Check problem formulation.")
