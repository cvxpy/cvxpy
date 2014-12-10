"""
Copyright 2013 Steven Diamond

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

class Shape(object):
    """ Represents the dimensions of a matrix.

    Attributes:
        rows: The number of rows.
        cols: The number of columns.
    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        super(Shape, self).__init__()

    @property
    def size(self):
        """Getter for (rows, cols)
        """
        return (self.rows, self.cols)

    def __add__(self, other):
        """Determines the shape of two matrices added together.

        The expression's sizes must match unless one is a scalar,
        in which case it is promoted to the size of the other.

        Args:
            self: The shape of the left-hand matrix.
            other: The shape of the right-hand matrix.

        Returns:
            The shape of the matrix sum.

        Raises:
            Error: Incompatible dimensions.
        """
        if self.size == (1, 1):
            return other
        elif other.size == (1, 1):
            return self
        elif self.size == other.size:
            return self
        else:
            raise ValueError("Incompatible dimensions %s %s" % (self, other))

    def __sub__(self, other):
        """Same as add.
        """
        return self + other

    def __mul__(self, other):
        """Determines the shape of two matrices multiplied together.

        The left-hand columns must match the right-hand rows, unless
        one side is a scalar.

        Args:
            self: The shape of the left-hand matrix.
            other: The shape of the right-hand matrix.

        Returns:
            The shape of the matrix product.

        Raises:
            Error: Incompatible dimensions.
        """
        if self.size == (1, 1):
            return other
        elif other.size == (1, 1):
            return self
        elif self.cols == other.rows:
            return Shape(self.rows, other.cols)
        else:
            raise ValueError("Incompatible dimensions %s %s" % (self, other))

    def __div__(self, other):
        """Determines the shape of a matrix divided by a scalar.

        Args:
            self: The shape of the left-hand matrix.
            other: The shape of the right-hand scalar.

        Returns:
            The shape of the matrix division.
        """
        return self

    def __truediv__(self, other):
        """Determines the shape of a matrix divided by a scalar.

        Args:
            self: The shape of the left-hand matrix.
            other: The shape of the right-hand scalar.

        Returns:
            The shape of the matrix division.
        """
        return self

    def __str__(self):
        return "(%s, %s)" % (self.rows, self.cols)

    def __repr__(self):
        return "Shape(%s, %s)" % (self.rows, self.cols)

