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

from .. import utilities as u

class NonlinearConstraint(object):
    """
    A nonlinear inequality constraint:
        f(x) <= 0
    where f is twice-differentiable.

    TODO: this may not be the best way to handle these constraints, but it is
    one of many (of course).
    """
    # f - a nonlinear function
    # x - the variables involved in the function
    def __init__(self, f, x):
        self.f = f
        self.x = x
        # The shape of x in f(x)
        cols = self.x[0].size[1]
        rows = sum(var.size[0] for var in self.x)
        self.x_shape = u.Shape(rows, cols)
        super(NonlinearConstraint, self).__init__()

    @property
    def size(self):
        return (self.f()[0],1)

    # Returns the variables involved in the function
    # in order, i.e. f(x) = f(vstack(variables))
    def variables(self):
        return self.x

    # Place x0 = f() in the vector of all variables.
    def place_x0(self, big_x, var_offsets, interface):
        m, x0 = self.f()
        offset = 0
        for var in self.variables():
            var_size = var.size[0]*var.size[1]
            var_x0 = x0[offset:offset+var_size]
            interface.block_add(big_x, var_x0, var_offsets[var.id],
                                0, var_size, 1)
            offset += var_size

    # Place Df in the gradient of all functions.
    def place_Df(self, big_Df, Df, var_offsets, vert_offset, interface):
        horiz_offset = 0
        for var in self.variables():
            var_size = var.size[0]*var.size[1]
            var_Df = Df[:, horiz_offset:horiz_offset+var_size]
            interface.block_add(big_Df, var_Df,
                                vert_offset, var_offsets[var.id],
                                self.size[0], var_size)
            horiz_offset += var_size

    # Place H in the Hessian of all functions.
    def place_H(self, big_H, H, var_offsets, interface):
        offset = 0
        for var in self.variables():
            var_size = var.size[0]*var.size[1]
            var_H = H[offset:offset+var_size, offset:offset+var_size]
            interface.block_add(big_H, var_H,
                                var_offsets[var.id], var_offsets[var.id],
                                var_size, var_size)
            offset += var_size

    # Extract the function variables from the vector x of all variables.
    def extract_variables(self, x, var_offsets, interface):
        local_x = interface.zeros(*self.x_shape.size)
        offset = 0
        for var in self.variables():
            var_size = var.size[0]*var.size[1]
            value = x[var_offsets[var.id]:var_offsets[var.id]+var_size]
            interface.block_add(local_x, value, offset, 0, var_size, 1)
            offset += var_size
        return local_x
