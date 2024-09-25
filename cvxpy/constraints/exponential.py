"""
Copyright 2013 Steven Diamond, 2022 - the CVXPY Authors

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
from __future__ import annotations

import warnings
from typing import List, Tuple, TypeVar

import numpy as np

from cvxpy.constraints.cones import Cone
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes

Expression = TypeVar('Expression')


class ExpCone(Cone):
    """A reformulated exponential cone constraint.

    Operates elementwise on :math:`x, y, z`.

    Original cone:

    .. math::

        K = \\{(x,y,z) \\mid y > 0, ye^{x/y} <= z\\}
            \\cup \\{(x,y,z) \\mid x \\leq 0, y = 0, z \\geq 0\\}

    Reformulated cone:

    .. math::

        K = \\{(x,y,z) \\mid y, z > 0, y\\log(y) + x \\leq y\\log(z)\\}
             \\cup \\{(x,y,z) \\mid x \\leq 0, y = 0, z \\geq 0\\}

    Parameters
    ----------
    x : Expression
        x in the exponential cone.
    y : Expression
        y in the exponential cone.
    z : Expression
        z in the exponential cone.
    """

    def __init__(self, x: Expression, y: Expression, z: Expression, constr_id=None) -> None:
        Expression = cvxtypes.expression()
        self.x = Expression.cast_to_const(x)
        self.y = Expression.cast_to_const(y)
        self.z = Expression.cast_to_const(z)
        args = [self.x, self.y, self.z]
        for val in args:
            if not (val.is_affine() and val.is_real()):
                raise ValueError('All arguments must be affine and real.')
        xs, ys, zs = self.x.shape, self.y.shape, self.z.shape
        if xs != ys or xs != zs:
            msg = ("All arguments must have the same shapes. Provided arguments have"
                   "shapes %s" % str((xs, ys, zs)))
            raise ValueError(msg)
        super(ExpCone, self).__init__(args, constr_id)

    def __str__(self) -> str:
        return "ExpCone(%s, %s, %s)" % (self.x, self.y, self.z)

    def __repr__(self) -> str:
        return "ExpCone(%s, %s, %s)" % (self.x, self.y, self.z)

    @property
    def residual(self):
        # TODO(akshayka): The projection should be implemented directly.
        from cvxpy import Minimize, Problem, Variable, hstack, norm2
        if self.x.value is None or self.y.value is None or self.z.value is None:
            return None
        x = Variable(self.x.shape)
        y = Variable(self.y.shape)
        z = Variable(self.z.shape)
        constr = [ExpCone(x, y, z)]
        obj = Minimize(norm2(hstack([x, y, z]) -
                             hstack([self.x.value, self.y.value, self.z.value])))
        problem = Problem(obj, constr)
        return problem.solve()

    @property
    def size(self) -> int:
        """The number of entries in the combined cones.
        """
        return 3 * self.num_cones()

    def num_cones(self):
        """The number of elementwise cones.
        """
        return self.x.size

    def as_quad_approx(self, m: int, k: int) -> RelEntrConeQuad:
        return RelEntrConeQuad(self.y, self.z, -self.x, m, k)

    def cone_sizes(self) -> List[int]:
        """The dimensions of the exponential cones.

        Returns
        -------
        list
            A list of the sizes of the elementwise cones.
        """
        return [3]*self.num_cones()

    def is_dcp(self, dpp: bool = False) -> bool:
        """An exponential constraint is DCP if each argument is affine.
        """
        if dpp:
            with scopes.dpp_scope():
                return all(arg.is_affine() for arg in self.args)
        return all(arg.is_affine() for arg in self.args)

    def is_dgp(self, dpp: bool = False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.is_dcp()

    @property
    def shape(self) -> Tuple[int, ...]:
        s = (3,) + self.x.shape
        return s

    def save_dual_value(self, value) -> None:
        # TODO(akshaya,SteveDiamond): verify that reshaping below works correctly
        value = np.reshape(value, (-1, 3))
        dv0 = np.reshape(value[:, 0], self.x.shape)
        dv1 = np.reshape(value[:, 1], self.y.shape)
        dv2 = np.reshape(value[:, 2], self.z.shape)
        self.dual_variables[0].save_value(dv0)
        self.dual_variables[1].save_value(dv1)
        self.dual_variables[2].save_value(dv2)

    def _dual_cone(self, *args):
        """Implements the dual cone of the exponential cone
        See Pg 85 of the MOSEK modelling cookbook for more information"""
        if args == ():
            return ExpCone(-self.dual_variables[1], -self.dual_variables[0],
                           np.exp(1) * self.dual_variables[2])
        else:
            # some assertions for verifying `args`
            args_shapes = [arg.shape for arg in args]
            instance_args_shapes = [arg.shape for arg in self.args]
            assert len(args) == len(self.args)
            assert args_shapes == instance_args_shapes
            return ExpCone(-args[1], -args[0],
                           np.exp(1)*args[2])


class RelEntrConeQuad(Cone):
    """An approximate construction of the scalar relative entropy cone

    Definition:

    .. math::

        K_{re}=\\text{cl}\\{(x,y,z)\\in\\mathbb{R}_{++}\\times
                \\mathbb{R}_{++}\\times\\mathbb{R}_{++}\\:x\\log(x/y)\\leq z\\}

    Since the above definition is very similar to the ExpCone, we provide a conversion method.

    More details on the approximation can be found in Theorem-3 on page-10 in the paper:
    Semidefinite Approximations of the Matrix Logarithm.

    Parameters
    ----------
    x : Expression
        x in the (approximate) scalar relative entropy cone
    y : Expression
        y in the (approximate) scalar relative entropy cone
    z : Expression
        z in the (approximate) scalar relative entropy cone
    m: Parameter directly related to the number of generated nodes for the quadrature
    approximation used in the algorithm
    k: Another parameter controlling the approximation
    """

    def __init__(self, x: Expression, y: Expression, z: Expression,
                 m: int, k: int, constr_id=None) -> None:
        Expression = cvxtypes.expression()
        self.x = Expression.cast_to_const(x)
        self.y = Expression.cast_to_const(y)
        self.z = Expression.cast_to_const(z)
        args = [self.x, self.y, self.z]
        for val in args:
            if not (val.is_affine() and val.is_real()):
                raise ValueError('All Expression arguments must be affine and real.')
        self.m = m
        self.k = k
        xs, ys, zs = self.x.shape, self.y.shape, self.z.shape
        if xs != ys or xs != zs:
            msg = ("All arguments must have the same shapes. Provided arguments have"
                   "shapes %s" % str((xs, ys, zs)))
            raise ValueError(msg)
        super(RelEntrConeQuad, self).__init__([self.x, self.y, self.z], constr_id)

    def get_data(self):
        return [self.m, self.k, self.id]

    def __str__(self) -> str:
        tup = (self.x, self.y, self.z, self.m, self.k)
        return "RelEntrConeQuad(%s, %s, %s, %s, %s)" % tup

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def residual(self):
        # TODO(akshayka): The projection should be implemented directly.
        from cvxpy import Minimize, Problem, Variable, hstack, norm2
        if self.x.value is None or self.y.value is None or self.z.value is None:
            return None
        cvxtypes.expression()
        x = Variable(self.x.shape)
        y = Variable(self.y.shape)
        z = Variable(self.z.shape)
        constr = [RelEntrConeQuad(x, y, z, self.m, self.k)]
        obj = Minimize(norm2(hstack([x, y, z]) -
                             hstack([self.x.value, self.y.value, self.z.value])))
        problem = Problem(obj, constr)
        return problem.solve()

    @property
    def size(self) -> int:
        """The number of entries in the combined cones.
        """
        return 3 * self.num_cones()

    def num_cones(self):
        """The number of elementwise cones.
        """
        return self.x.size

    def cone_sizes(self) -> List[int]:
        """The dimensions of the exponential cones.

        Returns
        -------
        list
            A list of the sizes of the elementwise cones.
        """
        return [3]*self.num_cones()

    def is_dcp(self, dpp: bool = False) -> bool:
        """An exponential constraint is DCP if each argument is affine.
        """
        if dpp:
            with scopes.dpp_scope():
                return all(arg.is_affine() for arg in self.args)
        return all(arg.is_affine() for arg in self.args)

    def is_dgp(self, dpp: bool = False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.is_dcp()

    @property
    def shape(self) -> Tuple[int, ...]:
        s = (3,) + self.x.shape
        return s

    def save_dual_value(self, value) -> None:
        # TODO: implement me.
        pass


class OpRelEntrConeQuad(Cone):
    """An approximate construction of the operator relative entropy cone

    Definition:

    .. math::

        K_{re}^n=\\text{cl}\\{(X,Y,T)\\in\\mathbb{H}^n_{++}\\times
                \\mathbb{H}^n_{++}\\times\\mathbb{H}^n_{++}\\:D_{\\text{op}}\\succeq T\\}

    More details on the approximation can be found in Theorem-3 on page-10 in the paper:
    Semidefinite Approximations of the Matrix Logarithm.

    Parameters
    ----------
    X : Expression
        x in the (approximate) operator relative entropy cone
    Y : Expression
        y in the (approximate) operator relative entropy cone
    Z : Expression
        Z in the (approximate) operator relative entropy cone
    m: int
        Must be positive. Controls the number of quadrature nodes used in a local
        approximation of the matrix logarithm. Increasing this value results in
        better local approximations, but does not significantly expand the region
        of inputs for which the approximation is effective.
    k: int
        Must be positive. Sets the number of scaling points about which the
        quadrature approximation is performed. Increasing this value will
        expand the region of inputs over which the approximation is effective.

    This approximation uses :math:`m + k` semidefinite constraints.
    """

    def __init__(self, X: Expression, Y: Expression, Z: Expression,
                 m: int, k: int, constr_id=None) -> None:
        Expression = cvxtypes.expression()
        self.X = Expression.cast_to_const(X)
        self.Y = Expression.cast_to_const(Y)
        self.Z = Expression.cast_to_const(Z)
        if (not X.is_hermitian()) or (not Y.is_hermitian()) or (not Z.is_hermitian()):
            msg = ("One of the input matrices has not explicitly been declared as symmetric or"
                   "Hermitian. If the inputs are Variable objects, try declaring them with the"
                   "symmetric=True or Hermitian=True properties. If the inputs are general "
                   "Expression objects that are known to be symmetric or Hermitian, then you"
                   "can wrap them with the symmetric_wrap and hermitian_wrap atoms. Failure to"
                   "do one of these things will cause this function to impose a symmetry or"
                   "conjugate-symmetry constraint internally, in a way that is very"
                   "inefficient.")
            warnings.warn(msg)
        self.m = m
        self.k = k
        Xs, Ys, Zs = self.X.shape, self.Y.shape, self.Z.shape
        if Xs != Ys or Xs != Zs:
            msg = ("All arguments must have the same shapes. Provided arguments have"
                   "shapes %s" % str((Xs, Ys, Zs)))
            raise ValueError(msg)
        super(OpRelEntrConeQuad, self).__init__([self.X, self.Y, self.Z], constr_id)

    def get_data(self):
        return [self.m, self.k, self.id]

    def __str__(self) -> str:
        tup = (self.X, self.Y, self.Z, self.m, self.k)
        return "OpRelEntrConeQuad(%s, %s, %s, %s, %s)" % tup

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def residual(self):
        # TODO: implement me
        raise NotImplementedError()

    @property
    def size(self) -> int:
        """The number of entries in the combined cones.
        """
        return 3 * self.num_cones()

    def num_cones(self):
        """The number of elementwise cones.
        """
        return self.X.size

    def cone_sizes(self) -> List[int]:
        """The dimensions of the exponential cones.

        Returns
        -------
        list
            A list of the sizes of the elementwise cones.
        """
        return [3]*self.num_cones()

    def is_dcp(self, dpp: bool = False) -> bool:
        """An operator relative conic constraint is DCP when (A, b, C) is affine
        """
        if dpp:
            with scopes.dpp_scope():
                return all(arg.is_affine() for arg in self.args)
        return all(arg.is_affine() for arg in self.args)

    def is_dgp(self, dpp: bool = False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.is_dcp()

    @property
    def shape(self) -> Tuple[int, ...]:
        s = (3,) + self.X.shape
        return s

    def save_dual_value(self, value) -> None:
        # TODO: implement me.
        pass
