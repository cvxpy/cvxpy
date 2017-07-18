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

from cvxpy.solver_interface.conic_solvers.ecos_conif import ECOS
from cvxpy.solver_interface.conic_solvers.ecos_bb_conif import ECOS_BB
from cvxpy.solver_interface.conic_solvers.cvxopt_conif import CVXOPT
from cvxpy.solver_interface.conic_solvers.glpk_conif import GLPK
from cvxpy.solver_interface.conic_solvers.glpk_mi_conif import GLPK_MI
from cvxpy.solver_interface.conic_solvers.cbc_conif import CBC
from cvxpy.solver_interface.conic_solvers.scs_conif import SCS
from cvxpy.solver_interface.conic_solvers.gurobi_conif import GUROBI
from cvxpy.solver_interface.conic_solvers.elemental_conif import Elemental
from cvxpy.solver_interface.conic_solvers.mosek_conif import MOSEK
from cvxpy.solver_interface.conic_solvers.ls_conif import LS
from cvxpy.solver_interface.conic_solvers.julia_opt_conif import JuliaOpt

solver_intf = [ECOS(), ECOS_BB(), CVXOPT(), GLPK(),
               GLPK_MI(), CBC(), SCS(), GUROBI(),
               Elemental(), MOSEK(), LS(), JuliaOpt()]
SOLVERS = {solver.name(): solver for solver in solver_intf}


def installed_solvers():
    """List the installed solvers.
    """
    installed = []
    for name, solver in SOLVERS.items():
        if solver.is_installed():
            installed.append(name)
    return installed
