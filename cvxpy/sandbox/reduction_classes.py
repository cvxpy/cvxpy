import numpy as np
import torch

import cvxpy as cp
from cvxpy.constraints import (
    Equality,
    Inequality,
    NonPos,
)
from cvxpy.reductions.utilities import (
    lower_equality,
    lower_ineq_to_nonneg,
    nonpos2nonneg,
)
from cvxtorch import TorchExpression


class HS071():
    def __init__(self, problem: cp.Problem):
        self.problem = problem
        # Assuming the problem has one main variable - adjust if needed
        self.main_var = []
        for var in self.problem.variables():
            self.main_var.append(var)
    
    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        # Set the variable value
        offset = 0
        for var in self.main_var:
            size = var.size
            var.value = x[offset:offset+size]
            offset += size
        # Evaluate the objective
        obj_value = self.problem.objective.args[0].value
        return obj_value
    
    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        # Convert to torch tensor with gradient tracking
        offset = 0
        torch_exprs = []
        for var in self.main_var:
            size = var.size
            slice = x[offset:offset+size]
            torch_exprs.append(torch.from_numpy(slice.astype(np.float64)).requires_grad_(True))
            offset += size
        
        # Create torch expression from CVXPY objective
        # Note: You'll need to implement TorchExpression.torch_expression properly
        # or use an alternative approach
        torch_obj = TorchExpression(self.problem.objective.args[0]).torch_expression(*torch_exprs)
        
        # Compute gradient
        torch_obj.backward()
        # Collect gradients properly
        gradients = []
        for tensor in torch_exprs:
            if tensor.grad is not None:
                gradients.append(tensor.grad.detach().numpy().flatten())
            else:
                # Handle case where gradient is None (shouldn't happen if requires_grad=True)
                gradients.append(np.zeros(tensor.numel()))
        
        # Concatenate all gradients
        gradient = np.concatenate(gradients)
        
        return gradient
    
    def constraints(self, x):
        """Returns the constraint values."""
        # Set the variable value
        offset = 0
        for var in self.main_var:
            size = var.size
            var.value = x[offset:offset+size]
            offset += size
        
        # Evaluate all constraints
        constraint_values = []
        for constraint in self.problem.constraints:
            constraint_values.append(constraint.args[0].value)
        return np.array(constraint_values)
    
    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        # Convert to torch tensor with gradient tracking
        offset = 0
        torch_vars_dict = {}
        torch_exprs = []
        
        for var in self.main_var:
            size = var.size
            slice = x[offset:offset+size]
            torch_tensor = torch.from_numpy(slice.astype(np.float64)).requires_grad_(True)
            torch_vars_dict[var.id] = torch_tensor  # Map CVXPY variable ID to torch tensor
            torch_exprs.append(torch_tensor)
            offset += size
        
        # Define a function that computes all constraint values
        def constraint_function(*args):
            # Create mapping from torch tensors back to CVXPY variables
            torch_to_var = {}
            for i, var in enumerate(self.main_var):
                torch_to_var[var.id] = args[i]
            
            constraint_values = []
            for constraint in self.problem.constraints:

                # Access the expression from the normalized constraint
                constraint_expr = constraint.args[0]
                
                # Get the variables that appear in this constraint expression
                constraint_vars = constraint_expr.variables()
                
                # Create ordered list of torch tensors for this specific constraint
                # in the order that the constraint expression expects them
                constr_torch_args = []
                for var in constraint_vars:
                    if var.id in torch_to_var:
                        constr_torch_args.append(torch_to_var[var.id])
                    else:
                        raise ValueError(f"Variable {var} not found in torch mapping")
                
                # Convert constraint expression to torch
                torch_expr = TorchExpression(constraint_expr).torch_expression(*constr_torch_args)
                constraint_values.append(torch_expr)
            
            if constraint_values:
                return torch.cat([torch.atleast_1d(cv) for cv in constraint_values])
            else:
                return torch.tensor([])
        
        # Compute Jacobian using torch.autograd.functional.jacobian
        if len(self.problem.constraints) > 0:
            jacobian_tuple = torch.autograd.functional.jacobian(constraint_function, 
                                                                tuple(torch_exprs))
            
            # Handle the case where jacobian_tuple is a tuple (multiple variables)
            if isinstance(jacobian_tuple, tuple):
                # Concatenate along the last dimension (variable dimension)
                jacobian_matrix = torch.cat(jacobian_tuple, dim=-1)
            else:
                # Single variable case
                jacobian_matrix = jacobian_tuple
            
            return jacobian_matrix.detach().numpy()
        else:
            # No constraints case
            total_size = sum(var.size for var in self.main_var)
            return np.array([]).reshape(0, total_size)

class Bounds_Getter():
    def __init__(self, problem: cp.Problem):
        self.problem = problem
        # Assuming the problem has one main variable - adjust if needed
        self.main_var = problem.variables()[0]
        self.get_constraint_bounds()
        self.get_variable_bounds()

    def get_constraint_bounds(self):
        "also normalizes the constraints"
        lower = []
        upper = []
        new_constr = []
        for constraint in self.problem.constraints:
            if isinstance(constraint, Equality):
                lower.append(np.zeros(constraint.size))
                upper.append(np.zeros(constraint.size))
                new_constr.append(lower_equality(constraint))
            elif isinstance(constraint, Inequality):
                lower.append(np.zeros(constraint.size))
                upper.append(np.inf * np.ones(constraint.size))
                new_constr.append(lower_ineq_to_nonneg(constraint))
            elif isinstance(constraint, NonPos):
                lower.append(np.zeros(constraint.size))
                upper.append(np.inf * np.ones(constraint.size))
                new_constr.append(nonpos2nonneg(constraint))
        
        lowered_con_problem = self.problem.copy([self.problem.objective, new_constr])
        self.new_problem = lowered_con_problem
        self.cl = lower
        self.cu = upper

    def get_variable_bounds(self):
        var_shape = self.main_var.size
        self.lb = np.ones(var_shape) * -np.inf
        self.ub = np.ones(var_shape) * np.inf
