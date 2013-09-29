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

# Utility functions to solve circular imports.
def constant():
    import constant
    return constant.Constant

def index_variable():
    import variables
    return variables.IndexVariable

def transpose_variable():
    import variables
    return variables.TransposeVariable

def variable():
    import variables
    return variables.Variable

def parameter():
    import parameter
    return parameter.Parameter

def expression():
    import expression
    return expression.Expression

def aff_obj():
    import affine
    return affine.AffObjective