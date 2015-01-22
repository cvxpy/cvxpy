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

class ProblemData(object):
    """A wrapper for the symbolic and numerical data for a problem.

    Attributes
    ----------
    sym_data : SymData
        The symbolic data for the problem.
    matrix_data : MatrixData
        The numerical data for the problem.
    prev_result : dict
        The result of the last solve.
    """
    def __init__(self):
        self.sym_data = None
        self.matrix_data = None
        self.prev_result = None
