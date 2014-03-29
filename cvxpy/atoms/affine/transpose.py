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

from affine_atom import AffAtom
from ... import utilities as u
from ... import interface as intf
from ...utilities import bool_mat_utils as bu
from ...utilities import coefficient_utils as cu
from ...expressions.variables import Variable
import numpy as np

class transpose(AffAtom):
    """ Matrix transpose. """
    # The string representation of the atom.
    def name(self):
        return "%s.T" % self.args[0]

    # Returns the transpose of the given value.
    @AffAtom.numpy_numeric
    def numeric(self, values):
        return values[0].T

    # Transposes shape, sign, and curvature.
    def init_dcp_attr(self):
        self._dcp_attr = self.args[0]._dcp_attr.T

    # Create a new variable equal to the argument transposed.
    def graph_implementation(self, arg_objs):
        # If arg_objs[0] is a Variable, no need to create a new variable.
        if isinstance(arg_objs[0], Variable):
            return super(transpose, self).graph_implementation(arg_objs)
        else:
            X = Variable(*arg_objs[0].size)
            constraints = [X == arg_objs[0]]
            return (X.T, constraints)

    def _tree_to_coeffs(self):
        """Create a coefficients dict for the transposed variable.

        Returns
        -------
        dict
            A dict of Variable to NumPy ndarray coefficient.
        """
        X = self.args[0]
        # The dimensions of the coefficients.
        cols = X.size[0]*X.size[1]
        rows = self.size[0]
        num_blocks = self.size[1]
        # Create cvxopt spmatrices to select the correct entries
        # from the vectorized X for each entry in the column.
        interface = intf.DEFAULT_SPARSE_INTERFACE
        blocks = []
        for k in xrange(num_blocks):
            # Convert to lil while constructing the matrix.
            coeff = interface.zeros(rows, cols).tolil()
            for i in xrange(rows):
                # Get the ith entry in row k of X.
                j = i*X.size[0] + k
                coeff[i, j] = 1
            blocks.append(coeff.tocsc())

        new_coeffs = {X.id: np.array(blocks, dtype="object", ndmin=1)}
        return cu.format_coeffs(new_coeffs)
