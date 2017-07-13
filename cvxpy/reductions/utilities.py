from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.qp2quad_form.qp_matrix_stuffing import QpMatrixStuffing
from cvxpy.reductions.qp2quad_form.qp2symbolic_qp import Qp2SymbolicQp
from cvxpy.solver_interface.conic_solvers import (ECOS, GUROBI, MOSEK, SCS,
                                                  Elemental, CBC, GLPK, CVXOPT)
from cvxpy.solver_interface.qp_solvers.qp_solver import QpSolver
from cvxpy.reductions.flip_objective import FlipObjective

REDUCTIONS = {ConeMatrixStuffing, Dcp2Cone, ECOS, Qp2SymbolicQp,
              QpMatrixStuffing, QpSolver, FlipObjective}
