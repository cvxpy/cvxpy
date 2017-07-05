"""
Copyright 2017 Robin Verschueren

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

import inspect
import sys
from cvxpy.constraints import Zero, NonPos, SOC, ExpCone, PSD


def attributes():
    """Return all attributes, i.e. all functions in this module except this function"""
    this_module_name = __name__
    return [obj for name, obj in inspect.getmembers(sys.modules[this_module_name])
            if (inspect.isfunction(obj) and
            name != 'attributes')]


def is_qp_constraint(constraint):
    if type(constraint) in {Zero, NonPos}:
        return True
    return False


def is_cone_constraint(constraint):
    if type(constraint) in {Zero, NonPos, SOC, ExpCone, PSD}:
        return True
    return False


def is_ecos_constraint(constraint):
    if type(constraint) in {Zero, NonPos, SOC, ExpCone}:
        return True
    return False


def are_arguments_affine(constraint):
    return all(arg.is_affine for arg in constraint.args)
