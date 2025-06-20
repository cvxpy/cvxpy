

import cvxpy as cp


def example_clnlbeam():
    # Problem parameters
    N = 1000
    h = 1 / N
    alpha = 350
    
    # Define variables with bounds
    t = cp.Variable(N+1)  # -1 <= t <= 1
    x = cp.Variable(N+1)  # -0.05 <= x <= 0.05
    u = cp.Variable(N+1)  # unbounded
    
    # Define objective function
    # Minimize: sum of 0.5*h*(u[i+1]^2 + u[i]^2) + 0.5*alpha*h*(cos(t[i+1]) + cos(t[i]))
    objective_terms = []
    for i in range(N):  # i from 0 to N-1 (Python 0-indexing)
        control_term = 0.5 * h * (u[i+1]**2 + u[i]**2)
        # Note: cos() is non-convex, this may cause solver issues
        trigonometric_term = 0.5 * alpha * h * (cp.cos(t[i+1]) + cp.cos(t[i]))
        objective_terms.append(control_term + trigonometric_term)
    
    objective = cp.Minimize(cp.sum(objective_terms))
    
    # Define constraints
    constraints = []
    
    # Variable bounds
    constraints.extend([
        t >= -1,
        t <= 1,
        x >= -0.05,
        x <= 0.05
    ])
    
    # Dynamics constraints
    for i in range(N):  # i from 0 to N-1
        # x[i+1] - x[i] - 0.5*h*(sin(t[i+1]) + sin(t[i])) == 0
        # Note: sin() is also non-convex
        position_constraint = (x[i+1] - x[i] - 
                             0.5 * h * (cp.sin(t[i+1]) + cp.sin(t[i])) == 0)
        constraints.append(position_constraint)
        
        # t[i+1] - t[i] - 0.5*h*u[i+1] - 0.5*h*u[i] == 0
        angle_constraint = (t[i+1] - t[i] - 
                          0.5 * h * u[i+1] - 0.5 * h * u[i] == 0)
        constraints.append(angle_constraint)
    
    # Create and solve the problem
    problem = cp.Problem(objective, constraints)
    return problem
