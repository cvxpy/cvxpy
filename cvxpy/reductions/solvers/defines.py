"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
from cvxpy.reductions.solvers.conic_solvers.super_scs_conif \
    import SuperSCS as SuperSCS_con
from cvxpy.reductions.solvers.conic_solvers.gurobi_conif \
    import GUROBI as GUROBI_con
from cvxpy.reductions.solvers.conic_solvers.xpress_conif \
    import XPRESS as XPRESS
from cvxpy.reductions.solvers.conic_solvers.mosek_conif \
    import MOSEK as MOSEK_con
from cvxpy.reductions.solvers.conic_solvers.cplex_conif \
    import CPLEX as CPLEX_con

# QP interfaces
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP as OSQP_qp
from cvxpy.reductions.solvers.qp_solvers.gurobi_qpif import GUROBI as GUROBI_qp
from cvxpy.reductions.solvers.qp_solvers.cplex_qpif import CPLEX as CPLEX_qp

solver_conic_intf = [ECOS_con(), ECOS_BB_con(),
                     CVXOPT_con(), GLPK_con(), XPRESS(),
                     GLPK_MI_con(), CBC_con(), SCS_con(), SuperSCS_con(),
                     GUROBI_con(), MOSEK_con(), CPLEX_con()]
solver_qp_intf = [OSQP_qp(),
                  GUROBI_qp(),
                  CPLEX_qp()
                  ]

SOLVER_MAP_CONIC = {solver.name(): solver for solver in solver_conic_intf}
SOLVER_MAP_QP = {solver.name(): solver for solver in solver_qp_intf}

# CONIC_SOLVERS and QP_SOLVERS are sorted in order of decreasing solver
# preference. QP_SOLVERS are those for which we have written interfaces
# and are supported by QpSolver.
CONIC_SOLVERS = [s.MOSEK, s.ECOS, s.SUPER_SCS, s.SCS,
                 s.CPLEX, s.GUROBI, s.GLPK, s.XPRESS,
                 s.GLPK_MI, s.CBC, s.CVXOPT, s.ECOS_BB]
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
    return np.unique(installed).tolist()


INSTALLED_SOLVERS = installed_solvers()
INSTALLED_CONIC_SOLVERS = [
  slv for slv in INSTALLED_SOLVERS if slv in CONIC_SOLVERS]
