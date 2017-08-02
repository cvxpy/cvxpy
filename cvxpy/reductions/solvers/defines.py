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

import cvxpy.settings as s
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import ECOS
from cvxpy.reductions.solvers.conic_solvers.ecos_bb_conif import ECOS_BB
from cvxpy.reductions.solvers.conic_solvers.cvxopt_conif import CVXOPT
from cvxpy.reductions.solvers.conic_solvers.glpk_conif import GLPK
from cvxpy.reductions.solvers.conic_solvers.glpk_mi_conif import GLPK_MI
from cvxpy.reductions.solvers.conic_solvers.cbc_conif import CBC
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.reductions.solvers.conic_solvers.gurobi_conif import GUROBI
from cvxpy.reductions.solvers.conic_solvers.elemental_conif import Elemental
from cvxpy.reductions.solvers.conic_solvers.mosek_conif import MOSEK
from cvxpy.reductions.solvers.conic_solvers.julia_opt_conif import JuliaOpt

solver_intf = [ECOS(), ECOS_BB(), CVXOPT(), GLPK(),
               GLPK_MI(), CBC(), SCS(), GUROBI(),
               Elemental(), MOSEK(), JuliaOpt()]
SOLVER_MAP = {solver.name(): solver for solver in solver_intf}

# CONIC_SOLVERS and QP_SOLVERS are sorted in order of decreasing solver
# preference. QP_SOLVERS are those for which we have written interfaces
# and are supported by QpSolver.
CONIC_SOLVERS = [s.MOSEK, s.ECOS, s.ECOS_BB, s.SCS, s.GUROBI, s.GLPK,
                 s.GLPK_MI, s.CBC, s.ELEMENTAL, s.JULIA_OPT, s.CVXOPT]
QP_SOLVERS = [s.MOSEK, s.GUROBI]


def installed_solvers():
    """List the installed solvers.
    """
    installed = []
    for name, solver in SOLVER_MAP.items():
        if solver.is_installed():
            installed.append(name)
    return installed


INSTALLED_SOLVERS = installed_solvers()
