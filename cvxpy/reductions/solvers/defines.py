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

from cvxpy.constraints import SOC

# Conic interfaces
from cvxpy.reductions.solvers.conic_solvers.cbc_conif import CBC as CBC_con
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL as CLARABEL_con
from cvxpy.reductions.solvers.conic_solvers.copt_conif import COPT as COPT_con
from cvxpy.reductions.solvers.conic_solvers.cosmo_conif import COSMO as COSMO_con
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
from cvxpy.reductions.solvers.conic_solvers.knitro_conif import KNITRO as KNITRO_con
from cvxpy.reductions.solvers.conic_solvers.moreau_conif import MOREAU as MOREAU_con
from cvxpy.reductions.solvers.conic_solvers.mosek_conif import MOSEK as MOSEK_con
from cvxpy.reductions.solvers.conic_solvers.nag_conif import NAG as NAG_con
from cvxpy.reductions.solvers.conic_solvers.pdcs_conif import PDCS as PDCS_con
from cvxpy.reductions.solvers.conic_solvers.pdlp_conif import PDLP as PDLP_con
from cvxpy.reductions.solvers.conic_solvers.qoco_conif import QOCO as QOCO_con
from cvxpy.reductions.solvers.conic_solvers.scip_conif import SCIP as SCIP_con
from cvxpy.reductions.solvers.conic_solvers.scipy_conif import SCIPY as SCIPY_con
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS as SCS_con
from cvxpy.reductions.solvers.conic_solvers.sdpa_conif import SDPA as SDPA_con
from cvxpy.reductions.solvers.conic_solvers.xpress_conif import XPRESS as XPRESS_con

# NLP interfaces
from cvxpy.reductions.solvers.nlp_solvers.copt_nlpif import COPT as COPT_nlp
from cvxpy.reductions.solvers.nlp_solvers.ipopt_nlpif import IPOPT as IPOPT_nlp
from cvxpy.reductions.solvers.nlp_solvers.uno_nlpif import UNO as UNO_nlp

# QP interfaces
from cvxpy.reductions.solvers.qp_solvers.copt_qpif import COPT as COPT_qp
from cvxpy.reductions.solvers.qp_solvers.cplex_qpif import CPLEX as CPLEX_qp
from cvxpy.reductions.solvers.qp_solvers.daqp_qpif import DAQP as DAQP_qp
from cvxpy.reductions.solvers.qp_solvers.gurobi_qpif import GUROBI as GUROBI_qp
from cvxpy.reductions.solvers.qp_solvers.highs_qpif import HIGHS as HIGHS_qp
from cvxpy.reductions.solvers.qp_solvers.knitro_qpif import KNITRO as KNITRO_qp
from cvxpy.reductions.solvers.qp_solvers.mpax_qpif import MPAX as MPAX_qp
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP as OSQP_qp
from cvxpy.reductions.solvers.qp_solvers.piqp_qpif import PIQP as PIQP_qp
from cvxpy.reductions.solvers.qp_solvers.proxqp_qpif import PROXQP as PROXQP_qp
from cvxpy.reductions.solvers.qp_solvers.qpalm_qpif import QPALM as QPALM_qp
from cvxpy.reductions.solvers.qp_solvers.xpress_qpif import XPRESS as XPRESS_qp

# Solver maps in preference order. Dict insertion order gives the preference
# ordering (previously maintained as separate CONIC_SOLVERS / QP_SOLVERS lists).
SOLVER_MAP_CONIC = {inst.name(): inst for inst in [
    MOSEK_con(), CLARABEL_con(), SCS_con(), ECOS_con(), MOREAU_con(),
    SDPA_con(), CPLEX_con(), GUROBI_con(), COPT_con(), GLPK_con(),
    NAG_con(), GLPK_MI_con(), CBC_con(), CVXOPT_con(), XPRESS_con(),
    DIFFCP_con(), SCIP_con(), SCIPY_con(), HIGHS_con(), GLOP_con(),
    PDLP_con(), QOCO_con(), CUCLARABEL_con(), CUOPT_con(), ECOS_BB_con(),
    KNITRO_con(), COSMO_con(), PDCS_con(),
]}

SOLVER_MAP_QP = {inst.name(): inst for inst in [
    OSQP_qp(), GUROBI_qp(), CPLEX_qp(), XPRESS_qp(), HIGHS_qp(),
    COPT_qp(), PIQP_qp(), PROXQP_qp(), QPALM_qp(), DAQP_qp(),
    MPAX_qp(), KNITRO_qp(),
]}

SOLVER_MAP_NLP = {inst.name(): inst for inst in [
    IPOPT_nlp(), UNO_nlp(), COPT_nlp(),
]}

# Preference-ordered solver name lists, derived from the maps above.
CONIC_SOLVERS = list(SOLVER_MAP_CONIC)
QP_SOLVERS = list(SOLVER_MAP_QP)
NLP_SOLVERS = list(SOLVER_MAP_NLP)

# Mixed-integer solver lists, derived from solver class attributes.
MI_SOLVERS = [
    name for name, slv in SOLVER_MAP_CONIC.items() if slv.MIP_CAPABLE
]
MI_SOCP_SOLVERS = [
    name for name, slv in SOLVER_MAP_CONIC.items()
    if slv.MIP_CAPABLE
    and SOC in getattr(slv, 'MI_SUPPORTED_CONSTRAINTS', slv.SUPPORTED_CONSTRAINTS)
]

# Policy list (not derivable from solver attributes).
COMMERCIAL_SOLVERS = [
    "MOSEK", "MOREAU", "GUROBI", "CPLEX", "COPT", "XPRESS", "NAG", "KNITRO",
]


def installed_solvers():
    """List the installed solvers."""
    return list(dict.fromkeys(
        name for name, slv in {**SOLVER_MAP_CONIC, **SOLVER_MAP_QP, **SOLVER_MAP_NLP}.items()
        if slv.is_installed()
    ))


# Installed solver lists.
INSTALLED_SOLVERS = installed_solvers()
INSTALLED_CONIC_SOLVERS = [slv for slv in INSTALLED_SOLVERS if slv in SOLVER_MAP_CONIC]
INSTALLED_MI_SOLVERS = [slv for slv in INSTALLED_SOLVERS if slv in MI_SOLVERS]
