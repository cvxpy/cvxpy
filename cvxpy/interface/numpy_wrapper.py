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

import numpy as np
from ..expressions import expression as exp
from .. import settings as s

# http://stackoverflow.com/questions/14619449/how-can-i-override-comparisons-between-numpys-ndarray-and-my-type

def override(name):
    """Wraps a Numpy comparison ufunc so cvxpy can overload the operator.

    Args:
        name: The name of a numpy comparison ufunc.

    Returns:
        A function.
    """
    # Numpy tries to convert the Expression to an array for ==.
    if name == "equal":
        def ufunc(x, y):
            if isinstance(y, np.ndarray) and y.ndim > 0 \
               and y[0] is s.NP_EQUAL_STR:
                    raise Exception("Prevent Numpy equal ufunc.")
            return getattr(np, name)(x, y)
        return ufunc
    else:
        def ufunc(x, y):
            if isinstance(y, exp.Expression):
                return NotImplemented
            return getattr(np, name)(x, y)
        return ufunc

# Implements operator overloading with comparisons.
np.set_numeric_ops(
    ** {
        ufunc : override(ufunc) for ufunc in (
            "less_equal", "equal", "greater_equal"
        )
    }
)
