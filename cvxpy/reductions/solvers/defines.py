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

import numpy as np
import scipy  # For version checks

import cvxpy.settings as s

# Conic interfaces
from cvxpy.reductions.solvers.conic_solvers.cbc_conif import CBC as CBC_con
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL as CLARABEL_con
from cvxpy.reductions.solvers.conic_solvers.copt_conif import COPT as COPT_con
from cvxpy.reductions.solvers.conic_solvers.cplex_conif import CPLEX as CPLEX_con
from cvxpy.reductions.solvers.conic_solvers.cuclarabel_conif import CUCLARABEL as CUCLARABEL_con
from cvxpy.reductions.solvers.conic_solvers.cuopt_conif import CUOPT as CUOPT_con
from cvxpy.reductions.solvers.conic_solvers.cvxopt_conif import CVXOPT as CVXOPT_con
from cvxpy.reductions.solvers.conic_solvers.diffcp_conif import DIFFCP as DIFFCP_con
from cvxpy.reductions.solvers.conic_solvers.ecos_bb_conif import ECOS_BB as ECOS_BB_con
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import ECOS as ECOS_con
from cvxpy.reductions.solvers.conic_solvers.glop_conif import GLOP as GLOP_con
from cvxpy.reductions.solvers.conic_solvers.glpk_conif import GLPK as GLPK_con
from cvxpy.reductions.solvers.conic_solvers.glpk_mi_conif import GLPK_MI as GLPK_MI_con
from cvxpy.reductions.solvers.conic_solvers.gurobi_conif import GUROBI as GUROBI_con
from cvxpy.reductions.solvers.conic_solvers.highs_conif import HIGHS as HIGHS_con
from cvxpy.reductions.solvers.conic_solvers.mosek_conif import MOSEK as MOSEK_con
from cvxpy.reductions.solvers.conic_solvers.nag_conif import NAG as NAG_con
from cvxpy.reductions.solvers.conic_solvers.pdlp_conif import PDLP as PDLP_con
from cvxpy.reductions.solvers.conic_solvers.qoco_conif import QOCO as QOCO_con
from cvxpy.reductions.solvers.conic_solvers.scip_conif import SCIP as SCIP_con
from cvxpy.reductions.solvers.conic_solvers.scipy_conif import SCIPY as SCIPY_con
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS as SCS_con
from cvxpy.reductions.solvers.conic_solvers.sdpa_conif import SDPA as SDPA_con
from cvxpy.reductions.solvers.conic_solvers.xpress_conif import XPRESS as XPRESS_con

# QP interfaces
from cvxpy.reductions.solvers.qp_solvers.copt_qpif import COPT as COPT_qp
from cvxpy.reductions.solvers.qp_solvers.cplex_qpif import CPLEX as CPLEX_qp
from cvxpy.reductions.solvers.qp_solvers.daqp_qpif import DAQP as DAQP_qp
from cvxpy.reductions.solvers.qp_solvers.gurobi_qpif import GUROBI as GUROBI_qp
from cvxpy.reductions.solvers.qp_solvers.highs_qpif import HIGHS as HIGHS_qp
from cvxpy.reductions.solvers.qp_solvers.mpax_qpif import MPAX as MPAX_qp
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP as OSQP_qp
from cvxpy.reductions.solvers.qp_solvers.piqp_qpif import PIQP as PIQP_qp
from cvxpy.reductions.solvers.qp_solvers.proxqp_qpif import PROXQP as PROXQP_qp
from cvxpy.reductions.solvers.qp_solvers.xpress_qpif import XPRESS as XPRESS_qp
from cvxpy.utilities.versioning import Version

solver_conic_intf = [DIFFCP_con(), ECOS_con(),
                     CVXOPT_con(), GLPK_con(), COPT_con(),
                     GLPK_MI_con(), CBC_con(), CLARABEL_con(), SCS_con(), SDPA_con(),
                     GUROBI_con(), MOSEK_con(), CPLEX_con(), NAG_con(), XPRESS_con(),
                     SCIP_con(), SCIPY_con(), HIGHS_con(), GLOP_con(), PDLP_con(),
                     QOCO_con(), CUCLARABEL_con(), CUOPT_con(), ECOS_BB_con()]

solver_qp_intf = [OSQP_qp(),
                  GUROBI_qp(),
                  CPLEX_qp(),
                  XPRESS_qp(),
                  COPT_qp(),
                  PIQP_qp(),
                  PROXQP_qp(),
                  DAQP_qp(),
                  HIGHS_qp(),
                  MPAX_qp(),
                  ]

SOLVER_MAP_CONIC = {solver.name(): solver for solver in solver_conic_intf}
SOLVER_MAP_QP = {solver.name(): solver for solver in solver_qp_intf}

# CONIC_SOLVERS and QP_SOLVERS are sorted in order of decreasing solver
# preference. QP_SOLVERS are those for which we have written interfaces
# and are supported by QpSolver.
CONIC_SOLVERS = [s.MOSEK, s.CLARABEL, s.SCS, s.ECOS, s.SDPA,
                 s.CPLEX, s.GUROBI, s.COPT, s.GLPK, s.NAG,
                 s.GLPK_MI, s.CBC, s.CVXOPT, s.XPRESS, s.DIFFCP,
                 s.SCIP, s.SCIPY, s.HIGHS, s.GLOP, s.PDLP, s.QOCO,
                 s.CUCLARABEL, s.CUOPT, s.ECOS_BB]

QP_SOLVERS = [s.OSQP,
              s.GUROBI,
              s.CPLEX,
              s.XPRESS,
              s.HIGHS,
              s.COPT,
              s.PIQP,
              s.PROXQP,
              s.DAQP,
              s.MPAX]
DISREGARD_CLARABEL_SDP_SUPPORT_FOR_DEFAULT_RESOLUTION = True
MI_SOLVERS = [s.GLPK_MI, s.MOSEK, s.GUROBI, s.CPLEX,
              s.XPRESS, s.CBC, s.SCIP, s.HIGHS, s.COPT, s.CUOPT, s.ECOS_BB]
MI_SOCP_SOLVERS = [s.MOSEK, s.GUROBI, s.CPLEX, s.XPRESS,
                   s.SCIP, s.ECOS_BB]

# Acknowledge MI solver support for SciPy >= 1.9.
if not (Version(scipy.__version__) < Version('1.9.0')):
    MI_SOLVERS.append(s.SCIPY)


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
INSTALLED_MI_SOLVERS = [
  slv for slv in INSTALLED_SOLVERS if slv in MI_SOLVERS]

