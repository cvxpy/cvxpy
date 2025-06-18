import cyipopt
import numpy as np
from reduction_classes import HS071, Bounds_Getter

import cvxpy as cp


def example_mle():

    # Generate data (matching Julia's Random.seed!(1234))
    n = 1000
    np.random.seed(1234)
    data = np.random.randn(n)
    
    # Define variables with starting values (CVXPY doesn't use start values the same way)
    mu = cp.Variable(1)  # mean parameter
    sigma = cp.Variable(shape=1)  # standard deviation, σ >= 0
    
    # Define the log-likelihood objective
    # Maximize: n/2 * log(1/(2*π*σ^2)) - sum((data[i] - μ)^2) / (2*σ^2)
    # This can be rewritten as:
    # Maximize: -n/2 * log(2*π) - n*log(σ) - sum((data[i] - μ)^2) / (2*σ^2)
    
    # The constant term -n/2 * log(2*π) can be dropped for optimization
    # So we maximize: -n*log(σ) - sum((data[i] - μ)^2) / (2*σ^2)
    
    # Calculate sum of squared residuals
    residual_sum = cp.sum(cp.square(data - mu))
    
    # Log-likelihood (without constant term)
    log_likelihood = -n * cp.log(sigma) - residual_sum / (2 * sigma**2)
    
    objective = cp.Maximize(log_likelihood)
    
    # Create and solve the problem
    problem = cp.Problem(objective)
    return problem

bounds = Bounds_Getter(example_mle())
x0 = [0.0, 1.0]

nlp = cyipopt.Problem(
   n=len(x0),
   m=len(bounds.cl),
   problem_obj=HS071(bounds.problem),
   lb=[None, 1e-6],
   ub=None,
   cl=bounds.cl,
   cu=bounds.cu,
)

nlp.add_option('mu_strategy', 'adaptive')
nlp.add_option('tol', 1e-7)
nlp.add_option('hessian_approximation', "limited-memory")
nlp.add_option('bound_relax_factor', 1e-8)
nlp.add_option('acceptable_tol', 1e-6)

x, info = nlp.solve(x0)
print(x)
