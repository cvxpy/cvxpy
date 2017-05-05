"""
Copyright 2013 Steven Diamond, 2017 Robin Verschueren

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

import numpy

from cvxpy import Maximize, Minimize, Problem
from cvxpy.atoms import *
from cvxpy.constraints import SOC, ExpCone
from cvxpy.error import SolverError
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variables import Bool, Semidef, Symmetric, Variable
from cvxpy.reductions.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.solver_interface.conic_solvers.ecos_conif import ECOS
from cvxpy.solver_interface.conic_solvers.gurobi_conif import GUROBI
from cvxpy.solver_interface.conic_solvers.mosek_conif import MOSEK
from cvxpy.solver_interface.conic_solvers.scs_conif import SCS
from cvxpy.solver_interface.lp_solvers.cbc_lpif import CBC
from cvxpy.reductions.dcp2qp import Dcp2Qp
from cvxpy.tests.base_test import BaseTest


class TestQp(BaseTest):
    """ Unit tests for the domain module. """

    def setUp(self):
        self.a = Variable(name='a')
        self.b = Variable(name='b')
        self.c = Variable(name='c')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable(2, 2, name='A')
        self.B = Variable(2, 2, name='B')
        self.C = Variable(3, 2, name='C')

    def test_qp(self):
        p = Problem(Minimize(quad_over_lin(norm1(self.x-1), 1)), [])
        self.assertTrue(Dcp2Qp().accepts(p))
        canon_p = Dcp2Qp().apply(p)
        pass
