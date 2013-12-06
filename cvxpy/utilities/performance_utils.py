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

# Taken from
# http://stackoverflow.com/questions/3012421/python-lazy-property-decorator

def lazyprop(func):
    """Wraps a property so it is lazily evaluated.

    Args:
        func: The property to wrap.

    Returns:
        A property that only does computation the first time it is called.
    """
    attr_name = '_lazy_' + func.__name__
    @property
    def _lazyprop(self):
        """A lazily evaluated propery.
        """
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    return _lazyprop
