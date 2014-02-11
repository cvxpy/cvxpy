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

import abc
import performance_utils as pu

class Affine(object):
    """
    An interface for objects that can be or can contain affine expressions.
    """

    __metaclass__ = abc.ABCMeta

    def coefficients(self, cache=False):
        """Returns a dict of Variable to coefficient.

        Args:
            cache: Should the coefficients be cached?

        Returns:
            A dict of Variable object to Numpy ndarray of column coefficients.
            Also includes the key settings.CONSTANT for constant terms.
        """
        # With no parameters, the coefficients can be cached.
        if cache and len(self.parameters()) == 0:
            return self._cached_coeffs
        # If there are parameters, the coefficients must be recalculated.
        else:
            return self._tree_to_coeffs()

    @pu.lazyprop
    def _cached_coeffs(self):
        """Caches the coefficients after the first call.

        Returns:
            A dict of Variable object to Numpy ndarray of column coefficients.
            Also includes the key settings.CONSTANT for constant terms.
        """
        return self._tree_to_coeffs()

    @abc.abstractmethod
    def _tree_to_coeffs(self):
        """Reduces the expression tree to a dict of terms to coefficients.

        Returns:
            A dict of Variable object to Numpy ndarray of column coefficients.
            Also includes the key settings.CONSTANT for constant terms.
        """
        return NotImplemented
