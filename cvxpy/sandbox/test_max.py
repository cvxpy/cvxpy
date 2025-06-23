import cyipopt
import numpy as np
from reduction_classes import HS071, Bounds_Getter

import cvxpy as cp


def example_max():
    # Define variables
    x = cp.Variable(1)
    y = cp.Variable(1)
    t = cp.Variable(1)
    
    objective = cp.Minimize(-t)
    
    constraints = [(t - x) * (t - y) == 0, t >= x, t >= y, x - 14 == 0, y - 6 == 0]
    
    problem = cp.Problem(objective, constraints)
    return problem

bounds = Bounds_Getter(example_max())
x0 = [12, 5, 0]

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


print(bounds.new_problem)
print(bounds.lb)
print(bounds.ub)
print(bounds.cl)
print(bounds.cu)
print(HS071(bounds.new_problem).constraints(np.array(x0)))
x, info = nlp.solve(x0)
print(x)
