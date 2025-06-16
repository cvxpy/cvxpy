import cyipopt
import numpy as np
import torch

import cvxpy as cp
from cvxtorch import TorchExpression


class HS071():
    def __init__(self, problem: cp.Problem):
        self.problem = problem
        # Assuming the problem has one main variable - adjust if needed
        self.main_var = problem.variables()[0]
    
    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        # Set the variable value
        self.main_var.value = x
        # Evaluate the objective
        obj_value = self.problem.objective.args[0].value
        print("objective:")
        print(obj_value)
        return obj_value
    
    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        # Convert to torch tensor with gradient tracking
        torch_x = torch.from_numpy(x.astype(np.float64)).requires_grad_(True)
        
        # Create torch expression from CVXPY objective
        # Note: You'll need to implement TorchExpression.torch_expression properly
        # or use an alternative approach
        torch_obj = TorchExpression(self.problem.objective.args[0]).torch_expression(torch_x)
        
        # Compute gradient
        torch_obj.backward()
        gradient = torch_x.grad.detach().numpy()
        print("gradient:")
        print(gradient)
        return gradient
    
    def constraints(self, x):
        """Returns the constraint values."""
        # Set the variable value
        self.main_var.value = x
        
        # Evaluate all constraints
        constraint_values = []
        for constraint in self.problem.constraints:
            if isinstance(constraint, cp.constraints.Inequality):
                constraint_values.append(constraint.args[1].value)
            elif isinstance(constraint, cp.constraints.Equality):
                constraint_values.append(constraint.args[0].value)
        
        print("constraints:")
        print(np.array(constraint_values))
        return np.array(constraint_values)
    
    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        # Convert to torch tensor with gradient tracking
        torch_x = torch.from_numpy(x.astype(np.float64)).requires_grad_(True)
        
        # Define a function that computes all constraint values
        def constraint_function(x_torch):
            constraint_values = []
            for constraint in self.problem.constraints:
                # Convert constraint expression to torch
                torch_expr = TorchExpression(constraint.expr).torch_expression(x_torch)

                constraint_values.append(torch_expr)
            
            if constraint_values:
                return torch.cat([torch.atleast_1d(cv) for cv in constraint_values])
            else:
                return torch.tensor([])
        
        # Compute Jacobian using torch.autograd.functional.jacobian
        if len(self.problem.constraints) > 0:
            jacobian_matrix = torch.autograd.functional.jacobian(constraint_function, torch_x)
        print("jacobian")
        print(jacobian_matrix.detach().numpy())
        return np.abs(jacobian_matrix.detach().numpy())

# Example usage for HS071 problem setup
def create_hs071_problem():
    """Creates the classic HS071 optimization problem."""
    # Variables
    x = cp.Variable(4)
    
    # Objective: minimize x[0]*x[3]*(x[0] + x[1] + x[2]) + x[2]
    objective = cp.Minimize(x[0]*x[3]*(x[0] + x[1] + x[2]) + x[2])
    
    # Constraints
    constraints = [
        x[0]*x[1]*x[2]*x[3] >= 25,  # Product constraint
        cp.sum_squares(x) == 40,    # Sum of squares constraint
    ]
    
    # Create problem
    problem = cp.Problem(objective, constraints)
    
    return problem

lb = [1.0, 1.0, 1.0, 1.0]
ub = [5.0, 5.0, 5.0, 5.0]

cl = [25.0, 40.0]
cu = [2.0e19, 40.0]

x0 = [1.0, 5.0, 5.0, 1.0]

nlp = cyipopt.Problem(
   n=len(x0),
   m=len(cl),
   problem_obj=HS071(create_hs071_problem()),
   lb=lb,
   ub=ub,
   cl=cl,
   cu=cu,
)

nlp.add_option('mu_strategy', 'adaptive')
nlp.add_option('tol', 1e-7)
nlp.add_option('hessian_approximation', "limited-memory")

x, info = nlp.solve(x0)
