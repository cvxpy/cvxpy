
import cyipopt
import numpy as np
from reduction_classes import HS071, Bounds_Getter

import cvxpy as cp


def example_qcp():
    # Define variables
    x = cp.Variable(1)
    y = cp.Variable(1, bounds=[0, np.inf])  # y >= 0
    z = cp.Variable(1, bounds=[0, np.inf])  # z >= 0
    
    # Define objective (maximize x)
    objective = cp.Minimize(-x)
    
    # Define constraints - exact same as JuMP model
    constraints = [
        x + y + z == 1,                # Linear equality constraint
        x**2 + y**2 - z**2 <= 0,      # Quadratic constraint: x*x + y*y - z*z <= 0
        x**2 - y*z <= 0               # Quadratic constraint: x*x - y*z <= 0
    ]
    
    # Create and solve problem
    problem = cp.Problem(objective, constraints)
    return problem

bounds = Bounds_Getter(example_qcp())
x0 = [0.2, 0.2, 0.2]

nlp = cyipopt.Problem(
   n=len(x0),
   m=len(bounds.cl),
   problem_obj=HS071(bounds.new_problem),
   lb=bounds.lb,
   ub=bounds.ub,
   cl=bounds.cl,
   cu=bounds.cu,
)

nlp.add_option('mu_strategy', 'adaptive')
nlp.add_option('tol', 1e-7)
nlp.add_option('hessian_approximation', "limited-memory")
nlp.add_option('print_level', 7)  # Increase for more detailed output

x, info = nlp.solve(x0)
print(x)
