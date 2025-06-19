import cyipopt
import numpy as np
from reduction_classes import HS071, Bounds_Getter

import cvxpy as cp


def example_mle():
    # Generate data (matching Julia's Random.seed!(1234))
    n = 1000
    np.random.seed(1234)
    data = np.random.randn(n)
    print(data.var(), data.mean())
    
    # Define variables - matching Julia exactly
    mu = cp.Variable(1, name="mu")  # mean parameter
    sigma = cp.Variable(1, name="sigma")  # standard deviation, σ >= 0
    
    constraints = [mu == sigma**2]
    
    # Calculate the objective exactly as in Julia
    # n / 2 * log(1 / (2 * π * σ^2)) - sum((data[i] - μ)^2 for i in 1:n) / (2 * σ^2)
    # = n/2 * log(1/(2*π)) - n*log(σ) - sum((data[i] - μ)^2) / (2*σ^2)
    
    # Sum of squared residuals
    residual_sum = cp.sum_squares(data - mu)
    
    # The complete objective (including the constant term that Julia includes)
    log_likelihood = (n / 2) * cp.log(1 / (2 * np.pi * (sigma)**2)) - residual_sum/(2 * (sigma)**2)
    
    objective = cp.Minimize(-log_likelihood)
    
    # Create problem with constraints
    problem = cp.Problem(objective, constraints)
    return problem

bounds = Bounds_Getter(example_mle())
x0 = [1.0, 0.0]

nlp = cyipopt.Problem(
   n=len(x0),
   m=len(bounds.cl),
   problem_obj=HS071(bounds.new_problem),
   lb=[1e-6, None],
   ub=None,
   cl=bounds.cl,
   cu=bounds.cu,
)

nlp.add_option('mu_strategy', 'adaptive')
nlp.add_option('tol', 1e-7)
nlp.add_option('hessian_approximation', "limited-memory")

x, info = nlp.solve(x0)
print(x)
