import numpy as np

import cvxpy as cp


def maxq_problem(n=20, bounds=None):
    """
    Solves the MAXQ problem:
    
    minimize max_{i=1..n} x_i^2
    subject to: l_i <= x_i <= u_i
    
    From the slides, the starting point is:
    x_i = i if i <= floor(n/2), else -i
    """
    # Set default bounds if not provided
    if bounds is None:
        # Create bounds based on the pattern in the slides
        n2 = n // 2
        lb = np.zeros(n)
        ub = np.zeros(n)
        
        for i in range(n):
            # Using reasonable bounds
            if i < n2:
                lb[i] = -10
                ub[i] = 10
            else:
                lb[i] = -10
                ub[i] = 10
    else:
        lb, ub = bounds
    
    # Variables
    x = cp.Variable(n)
    t = cp.Variable()  # For the max reformulation
    
    # Constraints
    constraints = []
    
    # Bound constraints
    constraints.append(x >= lb)
    constraints.append(x <= ub)
    
    # Max reformulation: t >= x_i^2 for all i
    for i in range(n):
        constraints.append(t >= cp.square(x[i]))
    
    # Objective: minimize t (which represents max(x_i^2))
    objective = cp.Minimize(t)
    
    # Problem
    prob = cp.Problem(objective, constraints)
    
    # Set initial values based on the pattern in slides
    n2 = n // 2
    x_init = np.zeros(n)
    for i in range(n):
        if i < n2:
            x_init[i] = i + 1
        else:
            x_init[i] = -(i + 1)
    
    # Note: CVXPY doesn't always use warm starts effectively,
    # but we can suggest starting values
    x.value = x_init
    
    # Solve
    prob.solve(verbose=True)
    
    print(f"Status: {prob.status}")
    print(f"Optimal value (max x_i^2): {prob.value}")
    print(f"Solution x: {x.value}")
    
    # Find which component achieves the maximum
    #_squared = x.value**2
    #max_idx = np.argmax(x_squared)
    
    return x.value, prob.value

# Alternative formulation without introducing auxiliary variable
def maxq_direct(n=20, bounds=None):
    """
    Direct formulation using cp.maximum
    """
    if bounds is None:
        lb = -10 * np.ones(n)
        ub = 10 * np.ones(n)
    else:
        lb, ub = bounds
    
    # Variables
    x = cp.Variable(n)
    
    # Constraints
    constraints = [x >= lb, x <= ub]
    
    # Objective: minimize max(x_i^2)
    # Create list of squared terms
    squared_terms = [cp.square(x[i]) for i in range(n)]
    
    # Use cp.maximum to find the maximum
    objective = cp.Minimize(cp.maximum(*squared_terms))
    
    # Problem
    prob = cp.Problem(objective, constraints)
    
    # Solve
    prob.solve(verbose=True)
    
    print("\nDirect formulation:")
    print(f"Status: {prob.status}")
    print(f"Optimal value: {prob.value}")
    
    return x.value, prob.value

# Example usage
if __name__ == "__main__":
    n = 20
    print(f"Solving MAXQ problem with n={n}")
    print("="*50)
    
    print("Method 1: With auxiliary variable t")
    x1, val1 = maxq_problem(n)
    
    print("\n" + "="*50)
    print("Method 2: Direct formulation")
    x2, val2 = maxq_direct(n)
    
    # Both methods should give the same result
    print(f"\nDifference in optimal values: {abs(val1 - val2)}")