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

from cvxpy.problems.solvers.ecos_intf import ECOS
from cvxpy.problems.solvers.ecos_bb_intf import ECOS_BB
from cvxpy.problems.solvers.cvxopt_intf import CVXOPT
from cvxpy.problems.solvers.glpk_intf import GLPK
from cvxpy.problems.solvers.glpk_mi_intf import GLPK_MI
from cvxpy.problems.solvers.scs_intf import SCS
from cvxpy.problems.solvers.gurobi_intf import GUROBI
from cvxpy.problems.solvers.elemental_intf import Elemental
from cvxpy.problems.solvers.mosek_intf import MOSEK

solver_intf = [ECOS(), ECOS_BB(), CVXOPT(), GLPK(),
               GLPK_MI(), SCS(), GUROBI(), Elemental(), MOSEK()]
SOLVERS = {solver.name():solver for solver in solver_intf}

def installed_solvers():
    """List the installed solvers.
    """
    installed = []
    for name, solver in SOLVERS.items():
        if solver.is_installed():
            installed.append(name)
    return installed
