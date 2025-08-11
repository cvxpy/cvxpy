import numpy as np

import cvxpy as cp


def chebyshev_rosenbrock(n=10):
    """
    Solves the Second Chebyshev-Rosenbrock problem:
    
    minimize (1/4)|x1 - 1| + sum_{i=1}^{n-1} |x_{i+1} - 2|x_i| + 1|
    
    The solution is x_i = 1 for all i.
    """
    # Variables
    x = cp.Variable(n)
    x.value = np.random.uniform(0, 1, n)  # Initial guess
    # Objective function
    obj = 0.25 * cp.abs(x[0] - 1)
    
    for i in range(n-1):
        # |x_{i+1} - 2|x_i| + 1|
        obj += cp.abs(x[i+1] - 2*cp.abs(x[i]) + 1)
    
    # Problem
    prob = cp.Problem(cp.Minimize(obj))
    
    # Solve
    prob.solve(verbose=True, solver=cp.IPOPT, nlp=True)
    
    print(f"Status: {prob.status}")
    print(f"Optimal value: {prob.value}")
    print(f"Solution x: {x.value}")
    
    # Check if solution is close to expected (x_i = 1 for all i)
    expected = np.ones(n)
    print(f"Distance from expected solution: {np.linalg.norm(x.value - expected)}")
    
    return x.value, prob.value

# Example usage
if __name__ == "__main__":
    # Test with n=10 (as in the slides)
    n = 10
    print(f"Solving Chebyshev-Rosenbrock problem with n={n}")
    solution, optimal_value = chebyshev_rosenbrock(n)
    
    # The slides mention the optimal value should be very close to 0
    print("\nExpected optimal value: ~0")
    print(f"Obtained optimal value: {optimal_value}")
