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

# Tests atoms by calling them with a constant value.
from cvxpy.settings import (SCS, ECOS, CVXOPT, GLPK, ELEMENTAL,
    OPTIMAL, ROBUST_KKTSOLVER)
from cvxpy.problems.solvers.utilities import installed_solvers
from cvxpy.atoms import *
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.problems.objective import *
from cvxpy.problems.problem import Problem
from cvxpy.problems.solvers.utilities import SOLVERS
from cvxpy.expressions.variables import Variable
from cvxpy.expressions.constants import Constant, Parameter
import cvxpy.interface as intf
import cvxopt
import numpy as np
import numpy.linalg as LA
import math
from nose.tools import assert_raises

ROBUST_CVXOPT = "robust_cvxopt"
SOLVER_TO_TOL = {SCS: 1e-2,
                 ECOS: 1e-7,
                 CVXOPT: 1e-7,
                 ROBUST_CVXOPT: 1e-7}
SOLVERS_TO_TRY = [ECOS, SCS, CVXOPT, ROBUST_CVXOPT]
# Test elemental if installed.
if ELEMENTAL in installed_solvers():
    SOLVERS_TO_TRY.append(ELEMENTAL)
    SOLVER_TO_TOL[ELEMENTAL] = 1e-7

v = cvxopt.matrix([-1,2,-2], tc='d')
v_np = np.matrix([-1.,2,-2]).T

# Atom, solver pairs known to fail.
KNOWN_SOLVER_ERRORS = []

atoms = [
    ([
        (abs, (2, 2), [ [[-5,2],[-3,1]] ],
            Constant([[5,2],[3,1]])),
        (diag, (2, 1), [ [[-5,2],[-3,1]] ], Constant([-5, 1])),
        (diag, (2, 2), [ [-5, 1] ], Constant([[-5, 0], [0, 1]])),
        (exp, (2, 2), [ [[1, 0],[2, -1]] ],
            Constant([[math.e, 1],[math.e**2, 1.0/math.e]])),
        (huber, (2, 2), [ [[0.5, -1.5],[4, 0]] ],
            Constant([[0.25, 2],[7, 0]])),
        (lambda x: huber(x, 2.5), (2, 2), [ [[0.5, -1.5],[4, 0]] ],
             Constant([[0.25, 2.25],[13.75, 0]])),
        (inv_pos, (2, 2), [ [[1,2],[3,4]] ],
            Constant([[1,1.0/2],[1.0/3,1.0/4]])),
        (lambda x: (x + Constant(0))**-1, (2, 2), [ [[1,2],[3,4]] ],
            Constant([[1,1.0/2],[1.0/3,1.0/4]])),
        (kl_div, (1, 1), [math.e, 1], Constant([1])),
        (kl_div, (1, 1), [math.e, math.e], Constant([0])),
        (lambda x: kron(np.matrix("1 2; 3 4"), x), (4, 4), [np.matrix("5 6; 7 8")],
            Constant(np.kron(np.matrix("1 2; 3 4").A, np.matrix("5 6; 7 8").A))),
        (lambda_max, (1, 1), [ [[2,0],[0,1]] ], Constant([2])),
        (lambda_max, (1, 1), [ [[5,7],[7,-3]] ], Constant([9.06225775])),
        (lambda x: lambda_sum_largest(x, 2), (1, 1), [ [[1, 2, 3], [2,4,5], [3,5,6]] ], Constant([11.51572947])),
        (log_sum_exp, (1, 1), [ [[5, 7], [0, -3]] ], Constant([7.1277708268])),
        (logistic, (2, 2),
         [
             [[math.log(5), math.log(7)],
              [0,           math.log(0.3)]] ],
         Constant(
             [[math.log(6), math.log(8)],
              [math.log(2), math.log(1.3)]])),
        (matrix_frac, (1, 1), [ [1, 2, 3],
                            [[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]] ], Constant([14])),
        (matrix_frac, (1, 1), [ [1, 2, 3],
                            [[67, 78, 90],
                             [78, 94, 108],
                             [90, 108, 127]] ], Constant([0.46557377049180271])),
        (max_elemwise, (2, 1), [ [-5,2],[-3,1],0,[-1,2] ], Constant([0,2])),
        (max_elemwise, (2, 2), [ [[-5,2],[-3,1]],0,[[5,4],[-1,2]] ],
            Constant([[5,4],[0,2]])),
        (max_entries, (1, 1), [ [[-5,2],[-3,1]] ], Constant([2])),
        (max_entries, (1, 1), [ [-5,-10] ], Constant([-5])),
        (lambda x: norm(x, 2), (1, 1), [v], Constant([3])),
        (lambda x: norm(x, "fro"), (1, 1), [ [[-1, 2],[3, -4]] ],
            Constant([5.47722557])),
        (lambda x: norm(x,1), (1, 1), [v], Constant([5])),
        (lambda x: norm(x,1), (1, 1), [ [[-1, 2], [3, -4]] ],
            Constant([10])),
        (lambda x: norm(x,"inf"), (1, 1), [v], Constant([2])),
        (lambda x: norm(x,"inf"), (1, 1), [ [[-1, 2], [3, -4]] ],
            Constant([4])),
        (lambda x: norm(x,"nuc"), (1, 1), [ [[2,0],[0,1]] ], Constant([3])),
        (lambda x: norm(x,"nuc"), (1, 1), [ [[3,4,5],[6,7,8],[9,10,11]] ],
            Constant([23.173260452512931])),
        (lambda x: norm(x,"nuc"), (1, 1), [ [[3,4,5],[6,7,8]] ],
            Constant([14.618376738088918])),
        (lambda x: sum_largest(abs(x), 3), (1, 1), [ [1,2,3,-4,-5] ], Constant([5+4+3])),
        (lambda x: mixed_norm(x,1,1), (1, 1), [ [[1,2],[3,4],[5,6]] ],
            Constant([21])),
        (lambda x: mixed_norm(x,1,1), (1, 1), [ [[1,2,3],[4,5,6]] ],
            Constant([21])),
        (lambda x: mixed_norm(x,2,1), (1, 1), [ [[3,3],[4,4]] ],
            Constant([10])),
        (lambda x: mixed_norm(x,1,'inf'), (1, 1), [ [[1,4],[5,6]] ],
            Constant([10])),

        (pnorm, (1, 1), [[1, 2, 3]], Constant([3.7416573867739413])),
        (lambda x: pnorm(x, 1), (1, 1), [[1.1, 2, -3]], Constant([6.1])),
        (lambda x: pnorm(x, 2), (1, 1), [[1.1, 2, -3]], Constant([3.7696153649941531])),
        (lambda x: pnorm(x, 'inf'), (1, 1), [[1.1, 2, -3]], Constant([3])),
        (lambda x: pnorm(x, 3), (1, 1), [[1.1, 2, -3]], Constant([3.3120161866074733])),
        (lambda x: pnorm(x, 5.6), (1, 1), [[1.1, 2, -3]], Constant([3.0548953718931089])),
        (lambda x: pnorm(x, 1.2), (1, 1), [[[1, 2, 3], [4, 5, 6]]], Constant([15.971021676279573])),

        (pos, (1, 1), [8], Constant([8])),
        (pos, (2, 1), [ [-3,2] ], Constant([0,2])),
        (neg, (2, 1), [ [-3,3] ], Constant([3,0])),

        (lambda x: power(x, 0), (1, 1), [7.45], Constant([1])),
        (lambda x: power(x, 1), (1, 1), [7.45], Constant([7.45])),
        (lambda x: power(x, 2), (1, 1), [7.45], Constant([55.502500000000005])),
        (lambda x: power(x, -1), (1, 1), [7.45], Constant([0.1342281879194631])),
        (lambda x: power(x, -.7), (1, 1), [7.45], Constant([0.24518314363015764])),
        (lambda x: power(x, -1.34), (1, 1), [7.45], Constant([0.06781263100321579])),
        (lambda x: power(x, 1.34), (1, 1), [7.45], Constant([14.746515290825071])),

        (quad_over_lin, (1, 1), [ [[-1,2,-2], [-1,2,-2]], 2], Constant([2*4.5])),
        (quad_over_lin, (1, 1), [v, 2], Constant([4.5])),
        (lambda x: norm(x, 2), (1, 1), [ [[2,0],[0,1]] ], Constant([2])),
        (lambda x: norm(x, 2), (1, 1), [ [[3,4,5],[6,7,8],[9,10,11]] ], Constant([22.368559552680377])),
        (lambda x: scalene(x, 2, 3), (2, 2), [ [[-5,2],[-3,1]] ], Constant([[15,4],[9,2]])),
        (square, (2, 2), [ [[-5,2],[-3,1]] ], Constant([[25,4],[9,1]])),
        (lambda x: (x + Constant(0))**2, (2, 2), [ [[-5,2],[-3,1]] ], Constant([[25,4],[9,1]])),
        (lambda x: sum_largest(x, 3), (1, 1), [ [1,2,3,4,5] ], Constant([5+4+3])),
        (lambda x: sum_largest(x, 3), (1, 1), [ [[3,4,5],[6,7,8],[9,10,11]] ], Constant([9+10+11])),
        (sum_squares, (1, 1), [ [[-1, 2],[3, -4]] ], Constant([30])),
        (trace, (1, 1), [ [[3,4,5],[6,7,8],[9,10,11]] ], Constant([3 + 7 + 11])),
        (trace, (1, 1), [ [[-5,2],[-3,1]]], Constant([-5 + 1])),
        (tv, (1, 1), [ [1,-1,2] ], Constant([5])),
        (tv, (1, 1), [ [[1],[-1],[2]] ], Constant([5])),
        (tv, (1, 1), [ [[-5,2],[-3,1]] ], Constant([math.sqrt(53)])),
        (tv, (1, 1), [ [[-5,2],[-3,1]], [[6,5],[-4,3]], [[8,0],[15,9]] ],
            Constant([LA.norm([7, -1, -8, 2, -10, 7])])),
        (tv, (1, 1), [ [[3,4,5],[6,7,8],[9,10,11]] ], Constant([4*math.sqrt(10)])),
        (upper_tri, (3, 1), [ [[3,4,5],[6,7,8],[9,10,11]] ], Constant([6, 9, 10])),
    ], Minimize),
    ([
        (entr, (2, 2), [ [[1, math.e],[math.e**2, 1.0/math.e]] ],
         Constant([[0, -math.e], [-2*math.e**2, 1.0/math.e]])),
        # #(entr(0), Constant([0])),
        (log_det, (1, 1),
               [ [[20, 8, 5, 2],
                  [8, 16, 2, 4],
                  [5, 2, 5, 2],
                  [2, 4, 2, 4]] ], Constant([7.7424020218157814])),
        (geo_mean, (1, 1), [[4, 1]], Constant([2])),
        (geo_mean, (1, 1), [[0.01, 7]], Constant([0.2645751311064591])),
        (geo_mean, (1, 1), [[63, 7]], Constant([21])),
        (geo_mean, (1, 1), [[1, 10]], Constant([math.sqrt(10)])),
        (lambda x: geo_mean(x, [1, 1]), (1, 1), [[1, 10]], Constant([math.sqrt(10)])),
        (lambda x: geo_mean(x, [.4, .8, 4.9]), (1, 1), [[.5, 1.8, 17]], Constant([10.04921378316062])),

        (harmonic_mean, (1, 1), [[1, 2, 3]], Constant([1.6363636363636365])),
        (harmonic_mean, (1, 1), [[2.5, 2.5, 2.5, 2.5]], Constant([2.5])),
        (harmonic_mean, (1, 1), [[0, 1, 2]], Constant([0])),

        (lambda x: diff(x, 0), (3, 1), [[1, 2, 3]], Constant([ 1,2,3 ])),
        (diff, (2, 1), [[1, 2, 3]], Constant([ 1,1 ])),
        (diff, (1, 1), [[1.1, 2.3]], Constant([1.2])),
        (lambda x: diff(x, 2), (1, 1), [[1, 2, 3]], Constant([ 0 ])),
        (diff, (3, 1), [[2.1, 1, 4.5, -.1]], Constant([ -1.1, 3.5, -4.6 ])),
        (lambda x: diff(x, 2), (2, 1), [[2.1, 1, 4.5, -.1]], Constant([ 4.6, -8.1 ])),


        (lambda x: pnorm(x, .5), (1, 1), [[1.1, 2, .1]], Constant([7.724231543909264])),
        (lambda x: pnorm(x, -.4), (1, 1), [[1.1, 2, .1]], Constant([0.02713620334])),
        (lambda x: pnorm(x, -1), (1, 1), [[1.1, 2, .1]], Constant([0.0876494023904])),
        (lambda x: pnorm(x, -2.3), (1, 1), [[1.1, 2, .1]], Constant([0.099781528576])),

        (lambda_min, (1, 1), [ [[2,0],[0,1]] ], Constant([1])),
        (lambda_min, (1, 1), [ [[5,7],[7,-3]] ], Constant([-7.06225775])),
        (lambda x: lambda_sum_smallest(x, 2), (1, 1), [ [[1, 2, 3], [2,4,5], [3,5,6]] ], Constant([-0.34481428])),
        (log, (2, 2), [ [[1, math.e],[math.e**2, 1.0/math.e]] ], Constant([[0, 1],[2, -1]])),
        (log1p, (2, 2), [ [[0, math.e-1],[math.e**2-1, 1.0/math.e-1]] ], Constant([[0, 1],[2, -1]])),
        (min_elemwise, (2, 1), [ [-5,2],[-3,1],0,[1,2] ], Constant([-5,0])),
        (min_elemwise, (2, 2), [ [[-5,2],[-3,-1]],0,[[5,4],[-1,2]] ], Constant([[-5,0],[-3,-1]])),
        (min_entries, (1, 1), [ [[-5,2],[-3,1]] ], Constant([-5])),
        (min_entries, (1, 1), [ [-5,-10] ], Constant([-10])),
        (lambda x: x**0.25, (1, 1), [7.45], Constant([7.45**0.25])),
        (lambda x: x**0.32, (2, 1), [ [7.45, 3.9] ], Constant(np.power(np.array([7.45, 3.9]), 0.32))),
        (lambda x: x**0.9, (2, 2),  [ [[7.45, 2.2], [4, 7]] ], Constant(np.power(np.array([[7.45, 2.2], [4, 7]]).T, 0.9))),
        (sqrt, (2, 2), [ [[2,4],[16,1]] ], Constant([[1.414213562373095,2],[4,1]])),
        (lambda x: sum_smallest(x, 3), (1, 1), [ [-1,2,3,4,5] ], Constant([-1+2+3])),
        (lambda x: sum_smallest(x, 4), (1, 1), [ [[-3,-4,5],[6,7,8],[9,10,11]] ], Constant([-3-4+5+6])),
        (lambda x: (x + Constant(0))**0.5, (2, 2), [ [[2,4],[16,1]] ], Constant([[1.414213562373095,2],[4,1]])),
    ], Maximize),
]

def check_solver(prob, solver_name):
    """Can the solver solve the problem?
    """
    objective, constraints = prob.canonicalize()
    if solver_name == ROBUST_CVXOPT:
        solver_name = CVXOPT
    solver = SOLVERS[solver_name]
    try:
        solver.validate_solver(constraints)
        return True
    except Exception as e:
        return False

# Tests numeric version of atoms.
def run_atom(atom, problem, obj_val, solver):
    assert problem.is_dcp()
    print(problem.objective)
    print(problem.constraints)
    if check_solver(problem, solver):
        print("solver", solver)
        tolerance = SOLVER_TO_TOL[solver]
        if solver == ROBUST_CVXOPT:
            result = problem.solve(solver=CVXOPT, verbose=False, kktsolver=ROBUST_KKTSOLVER)
        else:
            result = problem.solve(solver=solver, verbose=False)
        if problem.status is OPTIMAL:
            print(result)
            print(obj_val)
            assert( -tolerance <= (result - obj_val)/(1+np.abs(obj_val)) <= tolerance )
        else:
            assert (atom, solver) in KNOWN_SOLVER_ERRORS

def test_atom():
    for atom_list, objective_type in atoms:
        for atom, size, args, obj_val in atom_list:
            for row in range(size[0]):
                for col in range(size[1]):
                    for solver in SOLVERS_TO_TRY:
                        # Atoms with Constant arguments.
                        const_args = [Constant(arg) for arg in args]
                        yield (run_atom,
                               atom,
                               Problem(objective_type(atom(*const_args)[row,col])),
                               obj_val[row,col].value,
                               solver)
                        # Atoms with Variable arguments.
                        variables = []
                        constraints = []
                        for idx, expr in enumerate(args):
                            variables.append( Variable(*intf.size(expr) ))
                            constraints.append( variables[-1] == expr)
                        objective = objective_type(atom(*variables)[row,col])
                        yield (run_atom,
                               atom,
                               Problem(objective, constraints),
                               obj_val[row,col].value,
                               solver)
                        # Atoms with Parameter arguments.
                        parameters = []
                        for expr in args:
                            parameters.append( Parameter(*intf.size(expr)) )
                            parameters[-1].value = intf.DEFAULT_INTF.const_to_matrix(expr)
                        objective = objective_type(atom(*parameters)[row,col])
                        yield (run_atom,
                               atom,
                               Problem(objective),
                               obj_val[row,col].value,
                               solver)
