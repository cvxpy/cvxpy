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

from cvxpy.expressions.variables import Bool, Int
from cvxpy.problems.objective import Minimize
from cvxpy.atoms.affine_prod import affine_prod
from cvxpy.atoms.pnorm import pnorm
from cvxpy.atoms import abs, max_elemwise, sum_largest, max_entries, QuadForm


def attributes():
    """Return all attributes, i.e. all functions in this module except this function"""
    this_module_name = __name__
    return [obj for name, obj in inspect.getmembers(sys.modules[this_module_name])
            if (inspect.isfunction(obj) and
            name != 'attributes')]


def is_constrained(problem):
    return len(problem.constraints) != 0


def is_minimization(problem):
    return type(problem.objective) == Minimize


def is_dcp(problem):
    return problem.is_dcp()


def is_mixed_integer(problem):
    return any([isinstance(v, [Bool, Int]) for v in problem.variables()])


def has_pwl_atoms(problem):
    atom_types = [type(atom) for atom in problem.atoms()]
    pwl_types = [abs, affine_prod, max_elemwise, sum_largest, max_entries, pnorm]
    if any(atom in pwl_types for atom in atom_types):
        return True
    return False


def has_soc_atoms(problem):
    atom_types = [type(atom) for atom in problem.atoms()]
    soc_types = [QuadForm] # TODO
    if any(atom in soc_types for atom in atom_types):
        return True
    return False


def has_exp_atoms(problem):
    atom_types = [type(atom) for atom in problem.atoms()]
    exp_types = [] # TODO
    if any(atom in exp_types for atom in atom_types):
        return True
    return False


def has_psd_atoms(problem):
    atom_types = [type(atom) for atom in problem.atoms()]
    psd_types = [] # TODO
    if any(atom in psd_types for atom in atom_types):
        return True
    return False
