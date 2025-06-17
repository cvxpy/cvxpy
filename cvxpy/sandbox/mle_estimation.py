import numpy as np

import cvxpy as cp


def example_mle():

    # Generate data (matching Julia's Random.seed!(1234))
    n = 1000
    np.random.seed(1234)
    data = np.random.randn(n)
    
    # Define variables with starting values (CVXPY doesn't use start values the same way)
    μ = cp.Variable()  # mean parameter
    sigma = cp.Variable(pos=True)  # standard deviation, σ >= 0
    
    # Define the log-likelihood objective
    # Maximize: n/2 * log(1/(2*π*σ^2)) - sum((data[i] - μ)^2) / (2*σ^2)
    # This can be rewritten as:
    # Maximize: -n/2 * log(2*π) - n*log(σ) - sum((data[i] - μ)^2) / (2*σ^2)
    
    # The constant term -n/2 * log(2*π) can be dropped for optimization
    # So we maximize: -n*log(σ) - sum((data[i] - μ)^2) / (2*σ^2)
    
    # Calculate sum of squared residuals
    residual_sum = cp.sum(cp.square(data - μ))
    
    # Log-likelihood (without constant term)
    log_likelihood = -n * cp.log(sigma) - residual_sum / (2 * sigma**2)
    
    objective = cp.Maximize(log_likelihood)
    
    # Create and solve the problem
    problem = cp.Problem(objective)
    return problem
