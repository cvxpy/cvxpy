"""
Copyright 2017 Steven Diamond

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

from cvxpy.problems.solvers.ecos_intf import ECOS
from cvxpy.problems.solvers.ecos_bb_intf import ECOS_BB
from cvxpy.problems.solvers.cvxopt_intf import CVXOPT
from cvxpy.problems.solvers.glpk_intf import GLPK
from cvxpy.problems.solvers.glpk_mi_intf import GLPK_MI
from cvxpy.problems.solvers.cbc_intf import CBC
from cvxpy.problems.solvers.scs_intf import SCS
from cvxpy.problems.solvers.gurobi_intf import GUROBI
from cvxpy.problems.solvers.elemental_intf import Elemental
from cvxpy.problems.solvers.mosek_intf import MOSEK
from cvxpy.problems.solvers.ls_intf import LS
from cvxpy.problems.solvers.julia_opt_intf import JuliaOpt

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
