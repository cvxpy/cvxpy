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

import numbers

# A wrapper on the numeric function for norms
# so it can handle scalars.
def norm_numeric(old_numeric):
    def new_numeric(self, values):
        if isinstance(values[0], numbers.Number):
            return abs(values[0])
        else:
            return old_numeric(self, values)
    return new_numeric