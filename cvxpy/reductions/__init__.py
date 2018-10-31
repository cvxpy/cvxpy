"""
Copyright 2017 Robin Verschueren

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

from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution

from cvxpy.reductions.chain import Chain
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.complex2real.complex2real import Complex2Real
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.gp2dcp.gp2dcp import Gp2Dcp
from cvxpy.reductions.eval_params import EvalParams
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.qp2quad_form.qp_matrix_stuffing import QpMatrixStuffing
from cvxpy.reductions.qp2quad_form.qp2symbolic_qp import Qp2SymbolicQp
from cvxpy.reductions.solvers.solving_chain import SolvingChain
