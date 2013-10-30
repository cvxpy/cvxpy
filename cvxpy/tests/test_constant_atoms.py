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
from cvxpy.atoms import *
from cvxpy.problems.objective import *
from cvxpy.problems.problem import Problem
from cvxpy.expressions.variables import Variable
from cvxpy.expressions.constants import Constant, Parameter
import cvxopt
from nose.tools import assert_raises

TOL = 1e-3

v = cvxopt.matrix([-1,2,-2], tc='d')

atoms = [
    ([
        (abs([[-5,2],[-3,1]]), [5,2,3,1]),
        # #(huber(0.5), 0.25),
        # #(huber(-1.5), 2),
        # (inv_pos([[1,2],[3,4]]), [1,1.0/2,1.0/3,1.0/4]),
        # (lambda_max([[2,0],[0,1]]), [2]),
        # (lambda_max([[5,7],[7,-3]]), [9.06225775]),
        # (max([-5,2],[-3,1],0,[-1,2]), [0,2]),
        # (max([[-5,2],[-3,1]],0,[[5,4],[-1,2]]), [5,4,0,2]),
        # #(norm(v), 3),
        # (norm(v,2), [3]),
        # (norm([[-1, 2], [3, -4]], "fro"), [5.47722557]),
        # (norm(v,1), [5]),
        # (norm([[-1, 2], [3, -4]],1), [10]),
        # (norm(v,"inf"), [2]),
        # (norm([[-1, 2], [3, -4]],"inf"), [4]),
        # (norm([[2,0],[0,1]],"nuc"), [3]),
        # (norm([[3,4,5],[6,7,8],[9,10,11]],"nuc"), [23.1733]),
        # (pos(8), [8]),
        # (pos([-3,2]), [0,2]),
        # (neg([-3,3]), [3,0]),
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
        # (quad_over_lin(v, 2), [4.5]),
        # #(square_over_lin(2,4), 1),
        # (norm([[2,0],[0,1]]), [2]),
        # (norm([[3,4,5],[6,7,8],[9,10,11]]), [22.3686]),
        # (square([[-5,2],[-3,1]]), [25,4,9,1]),
    ], Minimize),
    ([
        # (geo_mean(4,1), [2]),
        # (geo_mean(2,2), [2]),
        # (lambda_min([[2,0],[0,1]]), [1]),
        # (lambda_min([[5,7],[7,-3]]), [-7.06225775]),
        # (min([-5,2],[-3,1],0,[1,2]), [-5,0]),
        # (min([[-5,2],[-3,-1]],0,[[5,4],[-1,2]]), [-5,0,-3,-1]),
        # #(pow_rat(4,1,2), 2),
        # #(pow_rat(8,1,3), 2),
        # #(pow_rat(16,1,4),2),
        # #(pow_rat(8,2,3), 4),
        # #(pow_rat(4,2,4), 2),
        # #(pow_rat(16,3,4),8),
        # (sqrt([[2,4],[16,1]]), [1.414213562373095,2,4,1]),
    ], Maximize),
]

# Tests numeric version of atoms.
def run_atom(problem, obj_val):
    assert problem.is_dcp()
    print problem.objective
    result = problem.solve()
    print result
    print obj_val
    assert( -TOL <= result - obj_val <= TOL )

def test_atom():
    for atom_list, objective_type in atoms:
        for atom, obj_val in atom_list:
            for counter, obj_index in enumerate(atom):
                # Atoms with Constant arguments.
                yield run_atom, Problem(objective_type(obj_index)), obj_val[counter]
                # Atoms with Variable arguments.
                variables = []
                constraints = []
                for exp in obj_index.subexpressions:
                    variables.append( Variable(*exp.size) )
                    constraints.append( variables[-1] == exp)
                atom = obj_index.__class__
                objective = objective_type(atom(*variables))
                yield run_atom, Problem(objective, constraints), obj_val[counter]
                # Atoms with Parameter arguments.
                # parameters = []
                # for exp in obj_index.subexpressions:
                #     parameters.append( Parameter(*exp.size) )
                #     parameters[-1].value = exp.value
                # atom(parameters[0])
                # objective = objective_type(atom(*parameters))
                # yield run_atom, Problem(objective), obj_val[counter]