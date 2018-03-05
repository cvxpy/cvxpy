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
import numpy as np

# Conic interfaces
from cvxpy.reductions.solvers.conic_solvers.ecos_conif \
    import ECOS as ECOS_con
from cvxpy.reductions.solvers.conic_solvers.ecos_bb_conif \
    import ECOS_BB as ECOS_BB_con
from cvxpy.reductions.solvers.conic_solvers.cvxopt_conif \
    import CVXOPT as CVXOPT_con
from cvxpy.reductions.solvers.conic_solvers.glpk_conif \
    import GLPK as GLPK_con
from cvxpy.reductions.solvers.conic_solvers.glpk_mi_conif \
    import GLPK_MI as GLPK_MI_con
from cvxpy.reductions.solvers.conic_solvers.cbc_conif \
    import CBC as CBC_con
from cvxpy.reductions.solvers.conic_solvers.scs_conif \
    import SCS as SCS_con
from cvxpy.reductions.solvers.conic_solvers.gurobi_conif \
    import GUROBI as GUROBI_con
from cvxpy.reductions.solvers.conic_solvers.xpress_conif \
    import XPRESS as XPRESS
from cvxpy.reductions.solvers.conic_solvers.elemental_conif \
    import Elemental as Elemental_con
from cvxpy.reductions.solvers.conic_solvers.mosek_conif \
    import MOSEK as MOSEK_con
from cvxpy.reductions.solvers.conic_solvers.julia_opt_conif \
    import JuliaOpt as JuliaOpt_con

# QP interfaces
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP as OSQP_qp
from cvxpy.reductions.solvers.qp_solvers.gurobi_qpif import GUROBI as GUROBI_qp
from cvxpy.reductions.solvers.qp_solvers.cplex_qpif import CPLEX as CPLEX_qp

solver_conic_intf = [ECOS_con(), ECOS_BB_con(),
                     CVXOPT_con(), GLPK_con(), XPRESS(),
                     GLPK_MI_con(), CBC_con(), SCS_con(), GUROBI_con(),
                     Elemental_con(), MOSEK_con(), JuliaOpt_con()]
solver_qp_intf = [OSQP_qp(),
                  GUROBI_qp(),
                  CPLEX_qp()
                  ]

SOLVER_MAP_CONIC = {solver.name(): solver for solver in solver_conic_intf}
SOLVER_MAP_QP = {solver.name(): solver for solver in solver_qp_intf}

# CONIC_SOLVERS and QP_SOLVERS are sorted in order of decreasing solver
# preference. QP_SOLVERS are those for which we have written interfaces
# and are supported by QpSolver.
CONIC_SOLVERS = [s.MOSEK, s.ECOS, s.ECOS_BB, s.SCS,
                 s.GUROBI, s.GLPK, s.XPRESS,
                 s.GLPK_MI, s.CBC, s.ELEMENTAL, s.JULIA_OPT, s.CVXOPT]
QP_SOLVERS = [s.OSQP,
              s.GUROBI,
              s.CPLEX]


def installed_solvers():
    """List the installed solvers.
    """
    installed = []
    # Check conic solvers
    for name, solver in SOLVER_MAP_CONIC.items():
        if solver.is_installed():
            installed.append(name)
    # Check QP solvers
    for name, solver in SOLVER_MAP_QP.items():
        if solver.is_installed():
            installed.append(name)

    # Remove duplicate names (for solvers that handle both conic and QP)
    np.unique(installed).tolist()

    return installed


INSTALLED_SOLVERS = installed_solvers()
