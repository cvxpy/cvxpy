"""
Copyright 2013 Steven Diamond

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

from __future__ import annotations

from cvxpy.reductions.reduction import Reduction


class QuadForm2SOC(Reduction):
    """Convert SymbolicQuadForm to SOC constraints.

    This reduction runs after Dcp2Cone and before ConeMatrixStuffing.
    It converts SymbolicQuadForm expressions (from quad_form, sum_squares, etc.)
    into second-order cone (SOC) constraints.

    Parameters
    ----------
    convert_objective : bool
        If True, convert SymbolicQuadForm in objective to SOC epigraph.
        If False, leave objective SymbolicQuadForm untouched (for solvers
        that natively support quadratic objectives).

    Notes
    -----
    Constraints always have their SymbolicQuadForm converted to SOC.

    The SOC form used is: ||[1-t, 2*u]||_2 <= 1+t  which gives ||u||^2 <= t

    For SymbolicQuadForm(x, P, expr) representing x.T @ P @ x:
    - Let L = MatrixSqrt(P), so P = L @ L.T
    - Then x.T @ P @ x = ||L.T @ x||^2
    - In objective: minimize t  s.t. ||L.T @ x||^2 <= t
    - In constraint: ||L.T @ x||^2 <= c becomes SOC constraint
    """

    def __init__(self, convert_objective: bool = True) -> None:
        super().__init__()
        self.convert_objective = convert_objective

    def accepts(self, problem):
        """This reduction accepts any problem."""
        return True

    def apply(self, problem):
        """Apply the reduction to convert SymbolicQuadForm to SOC.

        Parameters
        ----------
        problem : Problem
            The problem to transform.

        Returns
        -------
        Problem
            The transformed problem with SOC constraints.
        InverseData
            Data needed to invert the reduction.
        """
        # Import here to avoid circular imports
        from cvxpy.problems.objective import Minimize
        from cvxpy.problems.problem import Problem
        from cvxpy.reductions.inverse_data import InverseData

        inverse_data = InverseData(problem)
        new_constraints = []
        aux_vars = {}

        # Convert objective only if convert_objective=True
        if self.convert_objective:
            new_obj_expr = self._convert_expr(
                problem.objective.expr, new_constraints, aux_vars, in_objective=True
            )
        else:
            new_obj_expr = problem.objective.expr

        # Always convert constraints
        converted_constraints = []
        for constr in problem.constraints:
            new_constr = self._convert_constraint(constr, new_constraints, aux_vars)
            converted_constraints.append(new_constr)
            inverse_data.cons_id_map[constr.id] = new_constr.id

        # Store aux_vars for inversion
        inverse_data.aux_vars = aux_vars

        new_problem = Problem(
            Minimize(new_obj_expr),
            converted_constraints + new_constraints
        )
        return new_problem, inverse_data

    def _convert_expr(self, expr, new_constraints, aux_vars, in_objective=False):
        """Recursively convert SymbolicQuadForm nodes in an expression.

        Parameters
        ----------
        expr : Expression
            The expression to convert.
        new_constraints : list
            List to accumulate new SOC constraints.
        aux_vars : dict
            Dictionary mapping aux variable IDs to original expressions.
        in_objective : bool
            Whether this expression is in the objective.

        Returns
        -------
        Expression
            The converted expression.
        """
        # Import here to avoid circular imports
        from cvxpy.atoms.affine.hstack import hstack
        from cvxpy.atoms.quad_form import SymbolicQuadForm
        from cvxpy.constraints.second_order import SOC
        from cvxpy.expressions.constants.matrix_sqrt import MatrixSqrt
        from cvxpy.expressions.variable import Variable

        if isinstance(expr, SymbolicQuadForm):
            x = expr.args[0]
            P = expr.args[1]

            if expr.block_indices is not None:
                # Non-scalar SymbolicQuadForm: handle each block separately
                # Each block_indices[j] contains the input indices for output element j
                return self._convert_block_quad_form(
                    expr, x, P, new_constraints, aux_vars
                )

            # Scalar SymbolicQuadForm
            # Create symbolic matrix sqrt - factorization deferred to stuffing
            L = MatrixSqrt(P)
            # Flatten x first since P operates on flattened input
            x_flat = x.flatten(order='F')
            Lx = L.T @ x_flat

            # Epigraph formulation: replace with t, add ||Lx||^2 <= t
            # SOC form: ||[1-t, 2*Lx]||_2 <= 1+t
            # Note: don't use nonneg=True to match original quad_over_lin_canon
            t = Variable(1)
            aux_vars[t.id] = expr

            # Build SOC constraint
            Lx_flat = Lx.flatten(order='F')
            soc_X = hstack([1 - t, 2 * Lx_flat])
            new_constraints.append(SOC(t=1 + t, X=soc_X, axis=0))
            return t

        # Recursively convert args
        if hasattr(expr, 'args') and len(expr.args) > 0:
            new_args = [
                self._convert_expr(arg, new_constraints, aux_vars, in_objective)
                for arg in expr.args
            ]
            return expr.copy(new_args)

        # Leaf node - return as is
        return expr

    def _convert_block_quad_form(self, expr, x, P, new_constraints, aux_vars):
        """Convert non-scalar SymbolicQuadForm with block_indices to SOC constraints.

        For vectorized quad_over_lin (with axis parameter), each output element
        is a sum of squares over a subset of input elements. We create an SOC
        constraint for each output element.

        Parameters
        ----------
        expr : SymbolicQuadForm
            The symbolic quad form with block_indices.
        x : Expression
            The argument variable.
        P : Expression
            The P matrix (typically diagonal for block structures).
        new_constraints : list
            List to accumulate new SOC constraints.
        aux_vars : dict
            Dictionary mapping aux variable IDs to original expressions.

        Returns
        -------
        Expression
            A Variable with the same shape as the original output.
        """
        # Import here to avoid circular imports
        from cvxpy.atoms.affine.hstack import hstack
        from cvxpy.constraints.second_order import SOC
        from cvxpy.expressions.constants.matrix_sqrt import MatrixSqrt
        from cvxpy.expressions.variable import Variable

        block_indices = expr.block_indices
        n_outputs = len(block_indices)

        # Get the original expression shape from the SymbolicQuadForm
        output_shape = expr.original_expression.shape

        # Create output variable with same shape as original expression
        t = Variable(output_shape)
        aux_vars[t.id] = expr

        # Flatten x for indexing
        x_flat = x.flatten(order='F')

        # Create MatrixSqrt for the P matrix
        L = MatrixSqrt(P)

        # Flatten t for indexing
        t_flat = t.flatten(order='F')

        # For each output element, create an SOC constraint
        for j in range(n_outputs):
            indices = block_indices[j]

            # Get the subset of x for this block
            x_block = x_flat[indices]

            # Get the corresponding diagonal entries from L
            # Since P is diagonal for block structures, L is also diagonal
            # with L[i,i] = sqrt(P[i,i])
            Lx_block = L.T[indices, :][:, indices] @ x_block

            # Create SOC constraint: ||[1-t[j], 2*Lx_block]||_2 <= 1+t[j]
            t_j = t_flat[j]
            Lx_flat = Lx_block.flatten(order='F')
            soc_X = hstack([1 - t_j, 2 * Lx_flat])
            new_constraints.append(SOC(t=1 + t_j, X=soc_X, axis=0))

        return t

    def _convert_constraint(self, constr, new_constraints, aux_vars):
        """Convert SymbolicQuadForm in constraint args.

        Parameters
        ----------
        constr : Constraint
            The constraint to convert.
        new_constraints : list
            List to accumulate new SOC constraints.
        aux_vars : dict
            Dictionary mapping aux variable IDs to original expressions.

        Returns
        -------
        Constraint
            The converted constraint.
        """
        if hasattr(constr, 'args') and len(constr.args) > 0:
            new_args = [
                self._convert_expr(arg, new_constraints, aux_vars, in_objective=False)
                for arg in constr.args
            ]
            return constr.copy(new_args)
        return constr

    def invert(self, solution, inverse_data):
        """Invert the reduction to recover original solution.

        Parameters
        ----------
        solution : Solution
            The solution from the reduced problem.
        inverse_data : InverseData
            Data from apply() needed for inversion.

        Returns
        -------
        Solution
            The solution for the original problem.
        """
        # Remove auxiliary variables from primal_vars
        if hasattr(inverse_data, 'aux_vars') and solution.primal_vars is not None:
            for var_id in inverse_data.aux_vars:
                if var_id in solution.primal_vars:
                    del solution.primal_vars[var_id]
        return solution
