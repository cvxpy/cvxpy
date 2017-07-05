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

import numpy as np

from cvxpy.tests.base_test import BaseTest
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Minimize
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.expressions.variables.variable import Variable
from cvxpy.problems.problem_type import ProblemType
from cvxpy.expressions.attributes import is_affine, is_quadratic
from cvxpy.constraints import NonPos, Zero, PSD
from cvxpy.reductions.qp2quad_form.qp2symbolic_qp import Qp2SymbolicQp
from cvxpy.solver_interface.conic_solvers.ecos_conif import ECOS
from cvxpy.constraints.attributes import are_arguments_affine


class TestProblemType(BaseTest):
    """Unit tests for problem analysis and reduction path construction"""

    def setUp(self):
        self.x = Variable(2, name='x')
        self.Q = np.eye(2)
        self.c = np.array([1, 0.5])
        self.qp = Problem(Minimize(QuadForm(self.x, self.Q)), [self.x <= -1])
        self.cp = Problem(Minimize(self.c.T * self.x + 1), [self.x >= 0])

    def test_QPcanon_type(self):
        pa = ProblemType(self.qp)
        self.assertEquals(True, (Minimize, is_quadratic, True) in pa.type)
        self.assertEquals(True, (NonPos, is_affine, True) in pa.type)

    def test_QPcanon_standardQP_True(self):
        self.assertEquals(True, Qp2SymbolicQp().accepts(self.qp))

    def test_QPcanonaccepts_PSD_False(self):
        self.qp.constraints += [PSD(Variable(2, 2))]
        self.assertEquals(False, Qp2SymbolicQp().accepts(self.qp))

    def test_QPcanon_postconditions(self):
        pa = ProblemType(self.qp)
        pc = Qp2SymbolicQp.postconditions(pa.type)
        self.assertEquals(True, (Minimize, is_quadratic, True) in pc)
        self.assertEquals(True, (NonPos, are_arguments_affine, True) in pc)
        self.assertEquals(False, any(type(c[0]) == Zero for c in pc))

    def test_ECOSaccepts_standardCP_True(self):
        self.assertEquals(True, ECOS().accepts(self.cp))

    def test_ECOSaccepts_PSD_False(self):
        self.cp.constraints += [PSD(Variable(2, 2))]
        self.assertEquals(False, ECOS().accepts(self.cp))

    def test_atoms(self):
        atoms = self.qp.atoms()
        self.assertEquals(2, len(atoms))
