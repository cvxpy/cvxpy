
import cvxpy as cp


def example_qcp():
    # Define variables
    x = cp.Variable()
    y = cp.Variable(nonneg=True)  # y >= 0
    z = cp.Variable(nonneg=True)  # z >= 0
    
    # Define objective (maximize x)
    objective = cp.Maximize(x)
    
    # Define constraints - exact same as JuMP model
    constraints = [
        x + y + z == 1,                # Linear equality constraint
        x**2 + y**2 - z**2 <= 0,      # Quadratic constraint: x*x + y*y - z*z <= 0
        x**2 - y*z <= 0               # Quadratic constraint: x*x - y*z <= 0
    ]
    
    # Create and solve problem
    problem = cp.Problem(objective, constraints)
    return problem
