import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp

# Problem parameters
N = 100  # number of control intervals

# Decision variables
X = cp.Variable((2, N+1))  # state trajectory
pos = X[0, :]
speed = X[1, :]
U = cp.Variable((1, N))  # control trajectory (throttle)
T = cp.Variable(pos=True)  # final time (must be positive)

# Objective: minimize time
objective = cp.Minimize(T)

# Constraints list
constraints = []

# Time step (this creates non-convexity due to division by decision variable)
# We'll use dt = T/N in the dynamics constraints

# Dynamic constraints using RK4 integration
# dx/dt = f(x,u) where f(x,u) = [x[1]; u - x[1]]
for k in range(N):
    # Current state and control
    xk = X[:, k]
    uk = U[:, k]
    
    # RK4 integration
    # k1 = f(X(:,k), U(:,k))
    k1_1 = xk[1]
    k1_2 = uk[0] - xk[1]
    
    # k2 = f(X(:,k)+dt/2*k1, U(:,k))
    # Note: dt/2 * k1 creates bilinear terms (T * state/control)
    x_mid1_1 = xk[0] + (T/(2*N)) * k1_1
    x_mid1_2 = xk[1] + (T/(2*N)) * k1_2
    k2_1 = x_mid1_2
    k2_2 = uk[0] - x_mid1_2
    
    # k3 = f(X(:,k)+dt/2*k2, U(:,k))
    x_mid2_1 = xk[0] + (T/(2*N)) * k2_1
    x_mid2_2 = xk[1] + (T/(2*N)) * k2_2
    k3_1 = x_mid2_2
    k3_2 = uk[0] - x_mid2_2
    
    # k4 = f(X(:,k)+dt*k3, U(:,k))
    x_end_1 = xk[0] + (T/N) * k3_1
    x_end_2 = xk[1] + (T/N) * k3_2
    k4_1 = x_end_2
    k4_2 = uk[0] - x_end_2
    
    # x_next = X(:,k) + dt/6*(k1+2*k2+2*k3+k4)
    x_next_1 = xk[0] + (T/(6*N)) * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
    x_next_2 = xk[1] + (T/(6*N)) * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2)
    
    # Close the gap constraint
    constraints.append(X[0, k+1] == x_next_1)
    constraints.append(X[1, k+1] == x_next_2)

# Path constraints
# Speed limit: speed <= 1 - sin(2*pi*pos)/2
for i in range(N+1):
    # This is a non-convex constraint due to the sine function
    constraints.append(speed[i] <= 1 - cp.sin(2*np.pi*pos[i])/2)

# Control bounds
constraints.append(U >= 0)
constraints.append(U <= 1)

# Boundary conditions
constraints.append(pos[0] == 0)    # start at position 0
constraints.append(speed[0] == 0)  # start from standstill
constraints.append(pos[-1] == 1)   # finish line at position 1

# Time must be positive (already enforced by pos=True in variable declaration)
constraints.append(T >= 0.1)  # small lower bound for numerical stability

# Create problem
problem = cp.Problem(objective, constraints)

# Set initial values (warm start)
X.value = np.zeros((2, N+1))
X.value[1, :] = 1  # initial speed guess
U.value = 0.5 * np.ones((1, N))  # initial control guess
T.value = 1.0  # initial time guess

# Solve using IPOPT (or another NLP solver that CVXPY supports)
# Note: You need to have cvxpy-ipopt installed: pip install cvxpy-ipopt
result = problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
# Check if solution was found
if problem.status in ['optimal', 'optimal_inaccurate']:
    print("Optimal solution found!")
    print(f"Minimum time: {T.value:.4f} seconds")
    
    # Post-processing and visualization
    t = np.linspace(0, T.value, N+1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, speed.value, label='speed', linewidth=2)
    plt.plot(t, pos.value, label='position', linewidth=2)
    
    # Compute and plot speed limit
    speed_limit = 1 - np.sin(2*np.pi*pos.value)/2
    plt.plot(t, speed_limit, 'r--', label='speed limit', linewidth=2)
    
    # Plot control (throttle)
    plt.step(t[:-1], U.value.flatten(), 'k', where='post', label='throttle', linewidth=2)
    
    plt.xlabel('Time [s]')
    plt.ylabel('Value')
    plt.legend(loc='northwest')
    plt.grid(True, alpha=0.3)
    plt.title('Car Race Optimal Control Solution')
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"Final position: {pos.value[-1]:.4f}")
    print(f"Max speed reached: {np.max(speed.value):.4f}")
    print(f"Average throttle: {np.mean(U.value):.4f}")
    
else:
    print(f"Problem status: {problem.status}")
    print("Failed to find optimal solution")
