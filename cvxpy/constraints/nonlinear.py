"""
Copyright 2013 Steven Diamond, Eric Chu

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from cvxpy.constraints.constraint import Constraint


class NonlinearConstraint(Constraint):
    """
    A nonlinear inequality constraint:
        f(x) <= 0
    where f is twice-differentiable.

    TODO: this may not be the best way to handle these constraints, but it is
    one of many (of course).
    """
    # f - a nonlinear function
    # vars_ - the variables involved in the function

    def __init__(self, f, vars_):
        self.f = f
        self.vars_ = vars_
        # The shape of vars_ in f(vars_)
        cols = self.vars_[0].shape[1]
        rows = sum(var.shape[0] for var in self.vars_)
        self.x_shape = (rows*cols, 1)
        super(NonlinearConstraint, self).__init__(self.vars_)

    def variables(self):
        """Returns the variables involved in the function
           in order, i.e. f(vars_) = f(vstack(variables))
        """
        return self.vars_

    def block_add(self, matrix, block, vert_offset, horiz_offset, rows, cols,
                  vert_step=1, horiz_step=1):
        """Add the block to a slice of the matrix.

        Args:
            matrix: The matrix the block will be added to.
            block: The matrix/scalar to be added.
            vert_offset: The starting row for the matrix slice.
            horiz_offset: The starting column for the matrix slice.
            rows: The height of the block.
            cols: The width of the block.
            vert_step: The row step shape for the matrix slice.
            horiz_step: The column step shape for the matrix slice.
        """
        import cvxopt
        # Convert dense matrix to sparse if necessary.
        if isinstance(matrix, cvxopt.spmatrix) and isinstance(block, cvxopt.matrix):
            block = cvxopt.sparse(block)
        matrix[vert_offset:(rows+vert_offset):vert_step,
               horiz_offset:(horiz_offset+cols):horiz_step] += block

    def place_x0(self, big_x, var_offsets):
        """Place x0 = f() in the vector of all variables.
        """
        m, x0 = self.f()
        offset = 0
        for var in self.args:
            var_shape = var.shape[0]*var.shape[1]
            var_x0 = x0[offset:offset+var_shape]
            self.block_add(big_x, var_x0, var_offsets[var.data],
                           0, var_shape, 1)
            offset += var_shape

    def place_Df(self, big_Df, Df, var_offsets, vert_offset):
        """Place Df in the gradient of all functions.
        """
        horiz_offset = 0
        for var in self.args:
            var_shape = var.shape[0]*var.shape[1]
            var_Df = Df[:, horiz_offset:horiz_offset+var_shape]
            self.block_add(big_Df, var_Df,
                           vert_offset, var_offsets[var.data],
                           self.size, var_shape)
            horiz_offset += var_shape

    def place_H(self, big_H, H, var_offsets):
        """Place H in the Hessian of all functions.
        """
        offset = 0
        for var in self.args:
            var_shape = var.shape[0]*var.shape[1]
            var_H = H[offset:offset+var_shape, offset:offset+var_shape]
            self.block_add(big_H, var_H,
                           var_offsets[var.data], var_offsets[var.data],
                           var_shape, var_shape)
            offset += var_shape

    def extract_variables(self, x, var_offsets):
        """Extract the function variables from the vector x of all variables.
        """
        import cvxopt
        local_x = cvxopt.matrix(0., self.x_shape)
        offset = 0
        for var in self.args:
            var_shape = var.shape[0]*var.shape[1]
            value = x[var_offsets[var.data]:var_offsets[var.data]+var_shape]
            self.block_add(local_x, value, offset, 0, var_shape, 1)
            offset += var_shape
        return local_x
