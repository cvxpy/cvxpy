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

# Utility functions to solve circular imports.


def expression():
    from cvxpy.expressions import expression
    return expression.Expression


def add_expr():
    from cvxpy.atoms.affine import add_expr
    return add_expr.AddExpression


def conj():
    from cvxpy.atoms.affine import conj
    return conj.conj


def constant():
    from cvxpy.expressions import constants
    return constants.Constant


def variable():
    from cvxpy.expressions import variable
    return variable.Variable


def index():
    from cvxpy.atoms.affine import index
    return index.index


def special_index():
    from cvxpy.atoms.affine import index
    return index.special_index


def indicator():
    from cvxpy.transforms.indicator import indicator
    return indicator


def minimize():
    from cvxpy.problems import objective
    return objective.Minimize


def mul_expr():
    from cvxpy.atoms.affine import binary_operators
    return binary_operators.MulExpression


def multiply_expr():
    from cvxpy.atoms.affine import binary_operators
    return binary_operators.multiply


def div_expr():
    from cvxpy.atoms.affine import binary_operators
    return binary_operators.DivExpression


def neg_expr():
    from cvxpy.atoms.affine import unary_operators
    return unary_operators.NegExpression


def abs():
    from cvxpy.atoms.elementwise import abs
    return abs.abs


def lambda_min():
    from cvxpy.atoms import lambda_min
    return lambda_min


def pos():
    from cvxpy.atoms.elementwise import pos
    return pos.pos


def neg():
    from cvxpy.atoms.elementwise import neg
    return neg.neg


def partial_optimize():
    from cvxpy.transforms import partial_optimize
    return partial_optimize


def partial_problem():
    from cvxpy.transforms.partial_optimize import PartialProblem
    return PartialProblem


def power():
    from cvxpy.atoms.elementwise import power
    return power.power


def problem():
    from cvxpy.problems import problem
    return problem.Problem


def reshape():
    from cvxpy.atoms.affine import reshape
    return reshape.reshape


def transpose():
    from cvxpy.atoms.affine import transpose
    return transpose.transpose


def vec():
    from cvxpy.atoms.affine import vec
    return vec.vec


def vstack():
    from cvxpy.atoms.affine import vstack
    return vstack.vstack
