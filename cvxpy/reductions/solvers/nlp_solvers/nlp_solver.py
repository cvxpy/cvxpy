"""
Copyright 2025, the CVXPY developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

from cvxpy.constraints import (
    Equality,
    Inequality,
    NonPos,
)
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.reductions.utilities import (
    lower_equality,
    lower_ineq_to_nonneg,
    nonpos2nonneg,
)


class NLPsolver(Solver):
    """
    A non-linear programming (NLP) solver.
    """
    REQUIRES_CONSTR = False
    MIP_CAPABLE = False

    def accepts(self, problem):
        """
        Only accepts disciplined nonlinear programs.
        """
        return problem.is_dnlp()

    def apply(self, problem):
        """
        Construct NLP problem data stored in a dictionary.
        The NLP has the following form

            minimize      f(x)
            subject to    g^l <= g(x) <= g^u
                          x^l <= x <= x^u
        where f and g are non-linear (and possibly non-convex) functions
        """
        problem, data, inv_data = self._prepare_data_and_inv_data(problem)

        return data, inv_data

    def _prepare_data_and_inv_data(self, problem):
        data = dict()
        bounds = Bounds(problem)
        inverse_data = InverseData(bounds.new_problem)
        inverse_data.offset = 0.0
        data["problem"] = bounds.new_problem
        data["cl"], data["cu"] = bounds.cl, bounds.cu
        data["lb"], data["ub"] = bounds.lb, bounds.ub
        data["x0"] = bounds.x0
        data["_bounds"] = bounds  # Store for deferred Oracles creation in solve_via_data
        return problem, data, inverse_data

class Bounds():
    def __init__(self, problem):
        self.problem = problem
        self.main_var = problem.variables()
        self.get_constraint_bounds()
        self.get_variable_bounds()
        self.construct_initial_point()

    def get_constraint_bounds(self):
        """
        Get constraint bounds for all constraints.
        Also converts inequalities to nonneg form,
        as well as equalities to zero constraints and forms
        a new problem from the canonicalized problem.
        """
        lower, upper = [], []
        new_constr = []
        for constraint in self.problem.constraints:
            if isinstance(constraint, Equality):
                lower.extend([0.0] * constraint.size)
                upper.extend([0.0] * constraint.size)
                new_constr.append(lower_equality(constraint))
            elif isinstance(constraint, Inequality):
                lower.extend([0.0] * constraint.size)
                upper.extend([np.inf] * constraint.size)
                new_constr.append(lower_ineq_to_nonneg(constraint))
            elif isinstance(constraint, NonPos):
                lower.extend([0.0] * constraint.size)
                upper.extend([np.inf] * constraint.size)
                new_constr.append(nonpos2nonneg(constraint))
        canonicalized_prob = self.problem.copy([self.problem.objective, new_constr])
        self.new_problem = canonicalized_prob
        self.cl = np.array(lower)
        self.cu = np.array(upper)

    def get_variable_bounds(self):
        """
        Get variable bounds for all variables.
        Also takes into account nonneg/nonpos attributes.
        """
        var_lower, var_upper = [], []
        for var in self.main_var:
            size = var.size
            if var.bounds:
                lb = var.bounds[0].flatten(order='F')
                ub = var.bounds[1].flatten(order='F')
                if var.is_nonneg():
                    lb = np.maximum(lb, 0)
                if var.is_nonpos():
                    ub = np.minimum(ub, 0)
                var_lower.extend(lb)
                var_upper.extend(ub)
            else:
                # No bounds specified, use infinite bounds or bounds
                # set by the nonnegative or nonpositive attribute
                if var.is_nonneg():
                    var_lower.extend([0.0] * size)
                else:
                    var_lower.extend([-np.inf] * size)
                if var.is_nonpos():
                    var_upper.extend([0.0] * size)
                else:
                    var_upper.extend([np.inf] * size)
        self.lb = np.array(var_lower)
        self.ub = np.array(var_upper)
    

    def construct_initial_point(self):
        """ Loop through all variables and collect the intial point."""
        x0 = []
        for var in self.main_var:
            if var.value is None:
                raise ValueError("Variable %s has no value. This is a bug and should be reported."
                                  % var.name())

            x0.append(np.atleast_1d(var.value).flatten(order='F'))
        self.x0 = np.concatenate(x0, axis=0)

class Oracles():
    """
    Oracle interface for NLP solvers using the C-based diff engine.

    Provides function and derivative oracles (objective, gradient, constraints,
    jacobian, hessian) by wrapping the C_problem class from dnlp_diff_engine.
    """

    def __init__(self, problem, initial_point, num_constraints,
                 verbose: bool = True, use_hessian: bool = True):
        # Import from cvxpy's diff_engine integration layer
        from cvxpy.reductions.solvers.nlp_solvers.diff_engine import C_problem

        self.c_problem = C_problem(problem, verbose=verbose)
        self.use_hessian = use_hessian

        # Always initialize Jacobian
        self.c_problem.init_jacobian()

        # Only initialize Hessian if needed (not for quasi-Newton methods)
        if use_hessian:
            self.c_problem.init_hessian()

        self.initial_point = initial_point
        self.num_constraints = num_constraints
        self.iterations = 0

        # Cached sparsity structures
        self._jac_structure = None
        self._hess_structure = None
        self.constraints_forward_passed = False
        self.objective_forward_passed = False

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        self.objective_forward_passed = True
        return self.c_problem.objective_forward(x)

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        
        if not self.objective_forward_passed:
            self.objective(x)

        return self.c_problem.gradient()

    def constraints(self, x):
        """Returns the constraint values."""
        self.constraints_forward_passed = True
        return self.c_problem.constraint_forward(x)

    def jacobian(self, x):
        """Returns the Jacobian values in COO format at the sparsity structure. """

        if not self.constraints_forward_passed:
            self.constraints(x)

        jac_csr = self.c_problem.jacobian()
        jac_coo = jac_csr.tocoo()
        return jac_coo.data.copy()

    def jacobianstructure(self):
        """Returns the sparsity structure of the Jacobian."""
        if self._jac_structure is not None:
            return self._jac_structure

        jac_csr = self.c_problem.get_jacobian()
        jac_coo = jac_csr.tocoo()

        self._jac_structure = (jac_coo.row.astype(np.int32),
                               jac_coo.col.astype(np.int32))
        return self._jac_structure

    def hessian(self, x, duals, obj_factor):
        """Returns the lower triangular Hessian values in COO format. """
        if not self.use_hessian:
            # Shouldn't be called when using quasi-Newton, but return empty array
            return np.array([])

        if not self.objective_forward_passed:
            self.objective(x)
        if not self.constraints_forward_passed:
            self.constraints(x)

        hess_csr = self.c_problem.hessian(obj_factor, duals)
        hess_coo = hess_csr.tocoo()

        # Extract lower triangular values
        mask = hess_coo.row >= hess_coo.col

        return hess_coo.data[mask]

    def hessianstructure(self):
        """Returns the sparsity structure of the lower triangular Hessian."""
        if not self.use_hessian:
            # Return empty structure when using quasi-Newton approximation
            return (np.array([], dtype=np.int32), np.array([], dtype=np.int32))

        if self._hess_structure is not None:
            return self._hess_structure

        hess_csr = self.c_problem.get_hessian()
        hess_coo = hess_csr.tocoo()

        # Keep only lower triangular
        mask = hess_coo.row >= hess_coo.col
        self._hess_structure = (
            hess_coo.row[mask].astype(np.int32),
            hess_coo.col[mask].astype(np.int32)
        )
        return self._hess_structure

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""
        self.iterations = iter_count
        self.objective_forward_passed = False
        self.constraints_forward_passed = False


# TODO: maybe add a cchecker like this to the diff-engine? Or rather do a checker that
# uses cvxpy expressions to evaluate values. It will be slower, but will better test 
# consistency with cvxpy.
class DerivativeChecker:
    """
    A utility class to verify derivative computations by comparing
    C-based diff engine results against Python-based evaluations.
    """
    
    def __init__(self, problem):
        """
        Initialize the derivative checker with a CVXPY problem.
        
        Parameters
        ----------
        problem : cvxpy.Problem
            The CVXPY problem to check derivatives for.
        """
        from cvxpy.reductions.dnlp2smooth.dnlp2smooth import Dnlp2Smooth
        from cvxpy.reductions.solvers.nlp_solvers.diff_engine import C_problem
        
        self.original_problem = problem
        
        # Apply Dnlp2Smooth to get canonicalized problem
        canon = Dnlp2Smooth().apply(problem)
        self.canonicalized_problem = canon[0]
        
        # Construct the C version
        print("Constructing C diff engine problem for derivative checking...")
        self.c_problem = C_problem(self.canonicalized_problem)
        print("Done constructing C diff engine problem.")
        
        # Construct initial point using Bounds functionality
        self.bounds = Bounds(self.canonicalized_problem)
        self.x0 = self.bounds.x0
        
        # Initialize constraint bounds for checking
        self.cl = self.bounds.cl
        self.cu = self.bounds.cu
        
    def check_constraint_values(self, x=None):
        if x is None:
            x = self.x0
            
        # Evaluate constraints using C implementation
        c_values = self.c_problem.constraint_forward(x)
        
        # Evaluate constraints using Python implementation
        # First, set variable values
        x_offset = 0
        for var in self.canonicalized_problem.variables():
            var_size = var.size
            var.value = x[x_offset:x_offset + var_size].reshape(var.shape, order='F')
            x_offset += var_size
        
        # Now evaluate each constraint
        python_values = []
        for constr in self.canonicalized_problem.constraints:
            constr_val = constr.expr.value.flatten(order='F')
            python_values.append(constr_val)
        
        python_values = np.hstack(python_values) if python_values else np.array([])
        
        match = np.allclose(c_values, python_values, rtol=1e-10, atol=1e-10)        
        return match
    
    def check_jacobian(self, x=None, epsilon=1e-8): 
        if x is None:
            x = self.x0
        
        # Get Jacobian from C implementation
        self.c_problem.init_jacobian()
        self.c_problem.init_hessian()
        self.c_problem.constraint_forward(x) 
        c_jac_csr = self.c_problem.jacobian()
        c_jac_dense = c_jac_csr.toarray()
        
        # Compute numerical Jacobian using central differences
        n_vars = len(x)
        n_constraints = len(self.cl)
        numerical_jac = np.zeros((n_constraints, n_vars))
        
        # Define constraint function for finite differences
        def constraint_func(x_eval):
            return self.c_problem.constraint_forward(x_eval)
        
        # Compute each column using central differences
        for j in range(n_vars):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += epsilon
            x_minus[j] -= epsilon
            
            c_plus = constraint_func(x_plus)
            c_minus = constraint_func(x_minus)
            
            numerical_jac[:, j] = (c_plus - c_minus) / (2 * epsilon)
        
        match = np.allclose(c_jac_dense, numerical_jac, rtol=1e-4, atol=1e-5)
        return match
    
    def check_hessian(self, x=None, duals=None, obj_factor=1.0, epsilon=1e-8):
        if x is None:
            x = self.x0
        
        if duals is None:
            duals = np.random.rand(len(self.cl))
        
        # Get Hessian from C implementation
        self.c_problem.objective_forward(x)
        self.c_problem.constraint_forward(x)
        #jac = self.c_problem.jacobian()
        
        # must run gradient because for logistic it fills some values
        self.c_problem.gradient()
        c_hess_csr = self.c_problem.hessian(obj_factor, duals)
        
        # Convert to full dense matrix (C returns lower triangular)
        c_hess_coo = c_hess_csr.tocoo()
        n_vars = len(x)
        c_hess_dense = np.zeros((n_vars, n_vars))
        
        # Fill in the full symmetric matrix from lower triangular
        for i, j, v in zip(c_hess_coo.row, c_hess_coo.col, c_hess_coo.data):
            c_hess_dense[i, j] = v
            if i != j:
                c_hess_dense[j, i] = v
        
        # Compute numerical Hessian using finite differences of the Lagrangian gradient
        # Lagrangian gradient: ∇L = obj_factor * ∇f + J^T * duals
        def lagrangian_gradient(x_eval):
            self.c_problem.objective_forward(x_eval)
            grad_f = self.c_problem.gradient()
            
            self.c_problem.constraint_forward(x_eval)
            jac = self.c_problem.jacobian()
            
            # Lagrangian gradient = obj_factor * grad_f + J^T * duals
            return obj_factor * grad_f + jac.T @ duals
        
        # Compute Hessian via central differences of gradient
        numerical_hess = np.zeros((n_vars, n_vars))
        for j in range(n_vars):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += epsilon
            x_minus[j] -= epsilon
            
            grad_plus = lagrangian_gradient(x_plus)
            grad_minus = lagrangian_gradient(x_minus)
            
            numerical_hess[:, j] = (grad_plus - grad_minus) / (2 * epsilon)
        
        # Symmetrize the numerical Hessian (average with transpose to reduce numerical errors)
        numerical_hess = (numerical_hess + numerical_hess.T) / 2
        
        match = np.allclose(c_hess_dense, numerical_hess, rtol=1e-4, atol=1e-6)
        return match
    
    def check_objective_value(self, x=None):
        """ Compare objective value from C implementation with Python implementation. """
        print("Checking objective value...")
        if x is None:
            x = self.x0
        
        # Evaluate objective using C implementation
        c_obj_value = self.c_problem.objective_forward(x)
        
        # Evaluate objective using Python implementation
        x_offset = 0
        for var in self.canonicalized_problem.variables():
            var_size = var.size
            var.value = x[x_offset:x_offset + var_size].reshape(var.shape, order='F')
            x_offset += var_size
        
        python_obj_value = self.canonicalized_problem.objective.expr.value
        
        # Compare results
        match = np.allclose(c_obj_value, python_obj_value, rtol=1e-10, atol=1e-10)
        
        return match
    
    def check_gradient(self, x=None, epsilon=1e-8):
        """ Compare C-based gradient with numerical approximation using finite differences. """
        if x is None:
            x = self.x0
        print("Checking gradient...")
        # Get gradient from C implementation
        self.c_problem.objective_forward(x)
        c_grad = self.c_problem.gradient()
        
        # Compute numerical gradient using central differences
        n_vars = len(x)
        numerical_grad = np.zeros(n_vars)
        
        def objective_func(x_eval):
            return self.c_problem.objective_forward(x_eval)
        
        # Compute each component using central differences
        for j in range(n_vars):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += epsilon
            x_minus[j] -= epsilon
            
            f_plus = objective_func(x_plus)
            f_minus = objective_func(x_minus)
            
            numerical_grad[j] = (f_plus - f_minus) / (2 * epsilon)
            
        match = np.allclose(c_grad, numerical_grad, rtol= 5 * 1e-3, atol=1e-5)
        assert(match)
        return match
    
    def run(self, x=None):
        """ Run all derivative checks (constraints, Jacobian, and Hessian). """

        print("initializing derivatives for derivative checking...")
        self.c_problem.init_jacobian()
        self.c_problem.init_hessian()
        print("done initializing derivatives.")
        objective_result = self.check_objective_value(x)
        gradient_result = self.check_gradient(x)
        constraints_result = self.check_constraint_values()
        jacobian_result = self.check_jacobian(x)
        hessian_result = self.check_hessian(x)
       
        result = {'objective': objective_result,
                  'gradient': gradient_result,
                  'constraints': constraints_result,
                  'jacobian': jacobian_result,
                  'hessian': hessian_result}
        
        return result
    
    def run_and_assert(self, x=None):
        """ Run all derivative checks and assert correctness. """
        results = self.run(x)
        for key, passed in results.items():
            assert passed, f"Derivative check failed for {key}."