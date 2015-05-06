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

from cvxpy.atoms.atom import Atom
import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.elementwise.abs import abs
from cvxpy.atoms.affine.sum_entries import sum_entries
from numpy import linalg as LA
import numpy as np
from ..utilities.power_tools import sanitize_scalar, pow_high

# todo: OK, we've got a couple of (vector and matrix) norms here. maybe we dispatch to a pnorm (vector) atom

# todo: make sure (along with power and geo_mean) that we don't make extra variables and constraints if we don't need
# to, in the case where we have trivial powers like (0, 1, 0)

# def pnorm(x, p=2):
#
#     x = Expression.cast_to_const(x)
#     if p == 1:
#         return norm1(x)
#     elif p == "inf":
#         return normInf(x)
#     elif p == "nuc":
#         return normNuc(x)
#     elif p == "fro":
#         return norm2(x)
#     elif p == 2:
#         if x.is_matrix():
#             return sigma_max(x)
#         else:
#             return norm2(x)
#     else:
#         raise Exception("Invalid value %s for p." % p)


class pnorm(Atom):
    r"""Wrapper on the different norm atoms.

    .. math::

        \left(\sum_i |x_i|^p \right)^{1/p} \leq t


    The pnorm can be represented with the following inequalities

    .. math::

        |x_i| &\leq s_i^{1/p} t^{1 - 1/p}\\
        \sum_i s_i &\leq t,

    where :math:`p \geq 1`. (If we do it right, :math:`p = \infty` should also work.)

    The following are the special cases which we'd like to just automatically work.
    One thing to figure out is if we always need to form the extra variable for the absolute value term.

    - for :math:`p = 1`, we get

        .. math::

            |x_i| &\leq s_i^1 t^0\\
            \sum_i s_i &\leq t

      Note that we don't need the first inequality here, we just need the sum of absolute values

    - for :math:`p = \infty`

        .. math::

            |x_i| &\leq s_i^0 t^1\\
            \sum_i s_i &\leq t

      Here, we don't need the s variables, and we don't need the sum inequality

    - for :math:`p = 2`

        .. math::

            |x_i| &\leq s_i^{1/2} t^{1/2}\\
            \sum_i s_i &\leq t

      Since we can represent this natively, there's no need to do it this way. we can just use SOC,
      so this should be a special case.

      Although, its cool that we have this alternate decomposition. Maybe we should bake it in so
      that alternate decomps are always possible, just because its cool.


    Parameters
    ----------
    x : Expression or numeric constant
        The value to take the norm of.
    p : int, float, Fraction, or string
        The type of norm.

    Returns
    -------
    Expression
        An Expression representing the norm.
    """
    def __init__(self, x, p=2, max_denom=1024):
        if p in ('inf', 'Inf', np.inf):
            self.p, self.w = np.inf, None
        elif p < 1:
            raise ValueError("Norm must have p >= 1. {} is an invalid input.".format(p))
        else:
            p = sanitize_scalar(p)
            p, w = pow_high(p, max_denom)

        if p == 1:
            self.p, self.w = 1, None
        if p == 2:
            self.p, self.w = 2, None
        else:
            self.p, self.w = p, w

        super(pnorm, self).__init__(x)


    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the p-norm of x.
        """
        values = np.array(values[0]).flatten()
        return np.linalg.norm(values, self.p)


    def shape_from_args(self):
        """Resolves to a scalar.
        """
        return u.Shape(1, 1)

    def sign_from_args(self):
        """Always positive.
        """
        return u.Sign.POSITIVE

    def func_curvature(self):
        """Default curvature is convex.
        """
        return u.Curvature.CONVEX

    def monotonicity(self):
        """Increasing for positive arguments and decreasing for negative.
        """
        return [u.monotonicity.SIGNED]

    def get_data(self):
        return self.p, self.w

    def name(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.args[0].name(),
                               self.p)

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Reduces the atom to an affine expression and list of constraints.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        p, w = data

        if p == 1:
            pass
        elif p == 2:
            raise NotImplementedError('p={} is not yet implemented'.format(p))
        elif p == np.inf:
            raise NotImplementedError('p={} is not yet implemented'.format(p))
        else:
            raise NotImplementedError('p={} is not yet implemented'.format(p))

        # x = arg_objs[0]
        # obj, abs_constr = abs.graph_implementation([x], x.size)
        # obj, sum_constr = sum_entries.graph_implementation([obj], (1, 1))
        # return (obj, abs_constr + sum_constr)
