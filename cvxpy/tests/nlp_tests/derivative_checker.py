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

from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import Bounds


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

        match = np.allclose(c_grad, numerical_grad, rtol=5 * 1e-3, atol=1e-5)
        assert(match)
        return match

    def run(self, x=None):
        """ Run all derivative checks (constraints, Jacobian, and Hessian). """

        self.c_problem.init_jacobian()
        self.c_problem.init_hessian()
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
