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
from cvxpy.settings import SCS, SCS_MAT_FREE, ECOS, CVXOPT, OPTIMAL
from cvxpy.atoms import *
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.problems.objective import *
from cvxpy.problems.problem import Problem
from cvxpy.problems.solvers.utilities import SOLVERS
from cvxpy.expressions.variables import Variable
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.utilities.ordered_set import OrderedSet
import cvxpy.interface as intf
import cvxopt
import numpy.linalg as LA
import math
from nose.tools import assert_raises

SOLVER_TO_TOL = {SCS: 1e-1,
                 SCS_MAT_FREE: 1e-1,
                 ECOS: 1e-5,
                 CVXOPT: 1e-4}

v = cvxopt.matrix([-1,2,-2], tc='d')

# Atom, solver pairs known to fail.
KNOWN_SOLVER_ERRORS = [(lambda_min, SCS),
                       (lambda_max, SCS),
]

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
        (lambda_max, (1, 1), [ [[2,0],[0,1]] ], Constant([2])),
        (lambda_max, (1, 1), [ [[5,7],[7,-3]] ], Constant([9.06225775])),
        (log_sum_exp, (1, 1), [ [[5, 7], [0, -3]] ], Constant([7.1277708268])),
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
        # #(norm(v), 3),
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
        (lambda x: mixed_norm(x,1,1), (1, 1), [ [[1,2],[3,4],[5,6]] ],
            Constant([21])),
        (lambda x: mixed_norm(x,1,1), (1, 1), [ [[1,2,3],[4,5,6]] ],
            Constant([21])),
        (lambda x: mixed_norm(x,2,1), (1, 1), [ [[3,3],[4,4]] ],
            Constant([10])),
        (lambda x: mixed_norm(x,1,'inf'), (1, 1), [ [[1,4],[5,6]] ],
            Constant([10])),
        (pos, (1, 1), [8], Constant([8])),
        (pos, (2, 1), [ [-3,2] ], Constant([0,2])),
        (neg, (2, 1), [ [-3,3] ], Constant([3,0])),
        # #(pow_rat(4,1,1), 4),
        # #(pow_rat(2,2,1), 4),
        # #(pow_rat(4,2,2), 4),
        # #(pow_rat(2,3,1), 8),
        # #(pow_rat(4,3,2), 8),
        # #(pow_rat(4,3,3), 4),
        # #(pow_rat(2,4,1), 16),
        # #(pow_rat(4,4,2), 16),
        # #(pow_rat(8,4,3), 16),
        # #(pow_rat(8,4,4), 8),
        (quad_over_lin, (1, 1), [ [[-1,2,-2], [-1,2,-2]], 2], Constant([2*4.5])),
        (quad_over_lin, (1, 1), [v, 2], Constant([4.5])),
        # #(square_over_lin(2,4), 1),
        (lambda x: norm(x, 2), (1, 1), [ [[2,0],[0,1]] ], Constant([2])),
        (lambda x: norm(x, 2), (1, 1), [ [[3,4,5],[6,7,8],[9,10,11]] ], Constant([22.368559552680377])),
        (lambda x: scalene(x, 2, 3), (2, 2), [ [[-5,2],[-3,1]] ], Constant([[15,4],[9,2]])),
        (square, (2, 2), [ [[-5,2],[-3,1]] ], Constant([[25,4],[9,1]])),
        (lambda x: (x + Constant(0))**2, (2, 2), [ [[-5,2],[-3,1]] ], Constant([[25,4],[9,1]])),
        (sum_squares, (1, 1), [ [[-1, 2],[3, -4]] ], Constant([30])),
        (trace, (1, 1), [ [[3,4,5],[6,7,8],[9,10,11]] ], Constant([3 + 7 + 11])),
        (trace, (1, 1), [ [[-5,2],[-3,1]]], Constant([-5 + 1])),
        (tv, (1, 1), [ [1,-1,2] ], Constant([5])),
        (tv, (1, 1), [ [[1],[-1],[2]] ], Constant([5])),
        (tv, (1, 1), [ [[-5,2],[-3,1]] ], Constant([math.sqrt(53)])),
        (tv, (1, 1), [ [[-5,2],[-3,1]], [[6,5],[-4,3]], [[8,0],[15,9]] ],
            Constant([LA.norm([7, -1, -8, 2, -10, 7])])),
        (tv, (1, 1), [ [[3,4,5],[6,7,8],[9,10,11]] ], Constant([4*math.sqrt(10)])),
    ], Minimize),
    ([
        (entr, (2, 2), [ [[1, math.e],[math.e**2, 1.0/math.e]] ],
         Constant([[0, -math.e], [-2*math.e**2, 1.0/math.e]])),
        # #(entr(0), Constant([0])),
        (log_det, (1, 1),
               [ [[20, 8, 5, 2],
                  [8, 16, 2, 4],
                  [5, 2, 5, 2],
                  [2, 4, 2, 4]] ], Constant([7.7424])),
        (geo_mean, (1, 1), [4,1], Constant([2])),
        # (geo_mean, (1, 1), [0,7], Constant([0])),
        (geo_mean, (3, 2), [ [[2,63,3], [1,1,2]], [[8,7,3],[10,4,2]] ], Constant([[4,21,3],[math.sqrt(10),2,2]])),
        (lambda_min, (1, 1), [ [[2,0],[0,1]] ], Constant([1])),
        (lambda_min, (1, 1), [ [[5,7],[7,-3]] ], Constant([-7.06225775])),
        (log, (2, 2), [ [[1, math.e],[math.e**2, 1.0/math.e]] ], Constant([[0, 1],[2, -1]])),
        (log1p, (2, 2), [ [[0, math.e-1],[math.e**2-1, 1.0/math.e-1]] ], Constant([[0, 1],[2, -1]])),
        (min_elemwise, (2, 1), [ [-5,2],[-3,1],0,[1,2] ], Constant([-5,0])),
        (min_elemwise, (2, 2), [ [[-5,2],[-3,-1]],0,[[5,4],[-1,2]] ], Constant([[-5,0],[-3,-1]])),
        (min_entries, (1, 1), [ [[-5,2],[-3,1]] ], Constant([-5])),
        (min_entries, (1, 1), [ [-5,-10] ], Constant([-10])),
        # #(pow_rat(4,1,2), 2),
        # #(pow_rat(8,1,3), 2),
        # #(pow_rat(16,1,4),2),
        # #(pow_rat(8,2,3), 4),
        # #(pow_rat(4,2,4), 2),
        # #(pow_rat(16,3,4),8),
        (sqrt, (2, 2), [ [[2,4],[16,1]] ], Constant([[1.414213562373095,2],[4,1]])),
        (lambda x: (x + Constant(0))**0.5, (2, 2), [ [[2,4],[16,1]] ], Constant([[1.414213562373095,2],[4,1]])),
    ], Maximize),
]

def check_solver(prob, solver_name):
    """Can the solver solve the problem?
    """
    objective, constraints = prob.canonicalize()
    solver = SOLVERS[solver_name]
    try:
        solver.validate_solver(constraints)
        return True
    except Exception, e:
        return False

# Tests numeric version of atoms.
def run_atom(atom, problem, obj_val, solver):
    assert problem.is_dcp()
    print problem.objective
    print problem.constraints
    if check_solver(problem, solver):
        print "solver", solver
        tolerance = SOLVER_TO_TOL[solver]
        result = problem.solve(solver=solver)
        if problem.status is OPTIMAL:
            print result
            print obj_val
            assert( -tolerance <= result - obj_val <= tolerance )
        else:
            assert (atom, solver) in KNOWN_SOLVER_ERRORS

def test_atom():
    for atom_list, objective_type in atoms:
        for atom, size, args, obj_val in atom_list:
            for row in xrange(size[0]):
                for col in xrange(size[1]):
                    for solver in [ECOS, SCS, CVXOPT, SCS_MAT_FREE]:
                        # Atoms with Constant arguments.
                        yield (run_atom,
                               atom,
                               Problem(objective_type(atom(*args)[row,col])),
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
                            parameters[-1].value = intf.DEFAULT_INTERFACE.const_to_matrix(expr)
                        objective = objective_type(atom(*parameters)[row,col])
                        yield (run_atom,
                               atom,
                               Problem(objective),
                               obj_val[row,col].value,
                               solver)
