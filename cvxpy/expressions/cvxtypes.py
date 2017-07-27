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

# Utility functions to solve circular imports.


def add_expr():
    from cvxpy.atoms.affine import add_expr
    return add_expr.AddExpression


def constant():
    from cvxpy.expressions import constants
    return constants.Constant


def variable():
    from cvxpy.expressions import variables
    return variables.Variable


def index():
    from cvxpy.atoms.affine import index
    return index.index


def mul_expr():
    from cvxpy.atoms.affine import binary_operators
    return binary_operators.MulExpression


def rmul_expr():
    from cvxpy.atoms.affine import binary_operators
    return binary_operators.RMulExpression


def affine_prod_expr():
    from cvxpy.atoms import affine_prod
    return affine_prod


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


def power():
    from cvxpy.atoms.elementwise import power
    return power.power


def reshape():
    from cvxpy.atoms.affine import reshape
    return reshape.reshape


def transpose():
    from cvxpy.atoms.affine import transpose
    return transpose.transpose


def vec():
    from cvxpy.atoms.affine import vec
    return vec.vec
