import numpy as np
import cvxpy as cp

def griewank_function_cvxpy(d):
    """
    Implements the Griewank function using CVXPY.
    
    The Griewank function is defined as:
    f(x) = sum(x_i^2/4000) - prod(cos(x_i/sqrt(i))) + 1
    
    Parameters:
    d (int): Dimension of the problem
    
    Returns:
    x: CVXPY variable
    objective: CVXPY expression for the Griewank function
    Example taken from: https://www.sfu.ca/~ssurjano/griewank.html
    """
    
    # Define the optimization variable
    x = cp.Variable(d)
    
    # Define the Griewank function components
    # Sum term: sum(x_i^2 / 4000)
    sum_term = cp.sum([x[i]**2 for i in range(d)]) / 4000
    
    # Product term: prod(cos(x_i / sqrt(i+1)))
    # Start with the first cosine term
    prod_term = cp.cos(x[0] / cp.sqrt(1))
    
    # Multiply by remaining cosine terms
    for i in range(1, d):
        prod_term = prod_term * cp.cos(x[i] / cp.sqrt(i + 1))
    
    # Complete Griewank function
    objective = sum_term - prod_term + 1
    
    return x, objective


def create_griewank_problem(d):
    """
    Creates a complete Griewank optimization problem.
    
    Parameters:
    d (int): Dimension of the problem
    
    Returns:
    problem: CVXPY Problem object
    x: CVXPY variable
    """
    # Create the variable and objective
    x, objective = griewank_function_cvxpy(d)
    
    # Set up constraints (domain: x_i ∈ [-600, 600])
    constraints = [x >= -600, x <= 600]
    
    # Create the minimization problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    return problem, x


# Example usage
if __name__ == "__main__":
    # Set dimension
    d = 5
    
    # Create the Griewank optimization problem
    problem, x = create_griewank_problem(d)
    
    print("Griewank Function Optimization Problem")
    print("=" * 40)
    print(f"Dimensions: {d}")
    print("Domain: x_i ∈ [-600, 600] for all i")
    print("Global minimum: f(0, 0, ..., 0) = 0")
    print()
    x.value = np.random.uniform(-1, 1, d)  # Initial guess
    result = problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)

    if problem.status in ["optimal", "optimal_inaccurate"]:
        print(f"Solver status: {problem.status}")
        print(f"Objective value: {problem.value:.6f}")
        print(f"Solution x: {x.value}")
        print()
        
        # Verify with numpy implementation
        f_cvxpy = problem.value
    else:
        print(f"Solver status: {problem.status}")
        print("Note: The Griewank function is non-convex, which can cause")
        print("solver difficulties. You may need a global optimization solver.")
