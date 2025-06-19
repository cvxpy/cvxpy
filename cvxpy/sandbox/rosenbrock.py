import cyipopt
from reduction_classes import HS071, Bounds_Getter

import cvxpy as cp


def example_rosenbrock():
    # Define variables
    x = cp.Variable(1)
    y = cp.Variable(1)
    
    # Define objective: minimize (1 - x)^2 + 100 * (y - x^2)^2
    objective = cp.Minimize((1 - x)**2 + 100 * (y - x**2)**2)
    
    # No constraints for this problem
    constraints = []
    
    # Create and solve the problem
    problem = cp.Problem(objective, constraints)
    return problem

def example_rosenbrock_stacked():
    # Define variables
    x = cp.Variable(2)
    
    # Define objective: minimize (1 - x)^2 + 100 * (y - x^2)^2
    objective = cp.Minimize((1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2)
    
    # No constraints for this problem
    constraints = []
    
    # Create and solve the problem
    problem = cp.Problem(objective, constraints)
    return problem


bounds = Bounds_Getter(example_rosenbrock_stacked())
x0 = [0.0, 0.0]

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

x, info = nlp.solve(x0)
print(x)

# test normal version with two variables
bounds = Bounds_Getter(example_rosenbrock())
x0 = [0.0, 0.0]

nlp = cyipopt.Problem(
   n=len(x0),
   m=len(bounds.cl),
   problem_obj=HS071(bounds.new_problem),
   lb=None,
   ub=None,
   cl=bounds.cl,
   cu=bounds.cu,
)

nlp.add_option('mu_strategy', 'adaptive')
nlp.add_option('tol', 1e-7)
nlp.add_option('hessian_approximation', "limited-memory")

x, info = nlp.solve(x0)
print(x)
