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

from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from cvxpy.atoms.affine.promote import promote
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.errormsg import SECOND_ARG_SHOULD_NOT_BE_EXPRESSION_ERROR_MESSAGE
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import cvxtypes
from cvxpy.utilities.power_tools import (
    approx_error,
    decompose,
    fracify,
    lower_bound,
    over_bound,
    prettydict,
)


class geo_mean(Atom):
    """ The (weighted) geometric mean of vector ``x``, with optional powers given by ``p``:

    .. math::

        \\left(x_1^{p_1} \\cdots x_n^{p_n} \\right)^{\\frac{1}{\\mathbf{1}^Tp}}

    The powers ``p`` can be a ``list``, ``tuple``, or ``numpy.array`` of nonnegative
    ``int``, ``float``, or ``Fraction`` objects with nonzero sum.

    If not specified, ``p`` defaults to a vector of all ones, giving the
    **unweighted** geometric mean

    .. math::

        x_1^{1/n} \\cdots x_n^{1/n}.

    The geometric mean includes an implicit constraint that :math:`x_i \\geq 0`
    whenever :math:`p_i > 0`. If :math:`p_i = 0`, :math:`x_i` will be unconstrained.

    The only exception to this rule occurs when
    ``p`` has exactly one nonzero element, say, ``p_i``, in which case
    ``geo_mean(x, p)`` is equivalent to ``x_i`` (without the nonnegativity constraint).
    A specific case of this is when :math:`x \\in \\mathbf{R}^1`.


    .. note::

        Generally, ``p`` cannot be represented exactly, so a rational,
        i.e., fractional, **approximation** must be made.

        Internally, ``geo_mean`` immediately computes an approximate normalized
        weight vector :math:`w \\approx p/\\mathbf{1}^Tp`
        and the ``geo_mean`` atom is represented as

        .. math::

            x_1^{w_1} \\cdots x_n^{w_n},

        where the elements of ``w`` are ``Fraction`` objects that sum to exactly 1.

        The maximum denominator used in forming the rational approximation
        is given by ``max_denom``, which defaults to 1024, but can be adjusted
        to modify the accuracy of the approximations.

        The approximating ``w`` and the approximation error can be
        found through the attributes ``geo_mean.w`` and ``geo_mean.approx_error``.


    Examples
    --------

    The weights ``w`` can be seen from the string representation of the
    ``geo_mean`` object, or through the ``w`` attribute.

    >>> from cvxpy import Variable, geo_mean, Problem, Maximize
    >>> x = Variable(3, name='x')
    >>> print(geo_mean(x))
    geo_mean(x, (1/3, 1/3, 1/3))
    >>> g = geo_mean(x, [1, 2, 1])
    >>> g.w
    (Fraction(1, 4), Fraction(1, 2), Fraction(1, 4))

    Floating point numbers with few decimal places can sometimes be represented
    exactly. The approximation error between ``w`` and ``p/sum(p)`` is given by
    the ``approx_error`` attribute.

    >>> import numpy as np
    >>> x = Variable(4, name='x')
    >>> p = np.array([.12, .34, .56, .78])
    >>> g = geo_mean(x, p)
    >>> g.w
    (Fraction(1, 15), Fraction(17, 90), Fraction(14, 45), Fraction(13, 30))
    >>> g.approx_error
    0.0

    In general, the approximation is not exact.

    >>> p = [.123, .456, .789, .001]
    >>> g = geo_mean(x, p)
    >>> g.w
    (Fraction(23, 256), Fraction(341, 1024), Fraction(295, 512), Fraction(1, 1024))
    >>> 1e-4 <= g.approx_error <= 1e-3
    True

    The weight vector ``p`` can contain combinations of ``int``, ``float``,
    and ``Fraction`` objects.

    >>> from fractions import Fraction
    >>> x = Variable(4, name='x')
    >>> g = geo_mean(x, [.1, Fraction(1,3), 0, 2])
    >>> print(g)
    geo_mean(x, (3/73, 10/73, 0, 60/73))
    >>> g.approx_error <= 1e-10
    True

    Sequences of ``Fraction`` and ``int`` powers can often be represented **exactly**.

    >>> p = [Fraction(1,17), Fraction(4,9), Fraction(1,3), Fraction(25,153)]
    >>> x = Variable(4, name='x')
    >>> print(geo_mean(x, p))
    geo_mean(x, (1/17, 4/9, 1/3, 25/153))

    Terms with a zero power will not have an implicit nonnegativity constraint.

    >>> p = [1, 0, 1]
    >>> x = Variable(3, name='x')
    >>> obj = Maximize(geo_mean(x,p))
    >>> constr = [sum(x) <= 1, -1 <= x, x <= 1]
    >>> val = Problem(obj, constr).solve()
    >>> x = np.array(x.value).flatten()
    >>> print(x)
    [ 1. -1.  1.]


    Parameters
    ----------
    x : Variable
        A column or row vector whose elements we will take the geometric mean of.

    p : Sequence (list, tuple, ...) of ``int``, ``float``, or ``Fraction`` objects
        A vector of weights for the weighted geometric mean

        When ``p`` is a sequence of ``int`` and/or ``Fraction`` objects,
        ``w`` can often be an **exact** representation of the weights.
        An exact representation is sometimes possible when ``p`` has ``float``
        elements with only a few decimal places.

    max_denom : int
        The maximum denominator to use in approximating ``p/sum(p)`` with
        ``geo_mean.w``. If ``w`` is not an exact representation, increasing
        ``max_denom`` **may** offer a more accurate representation, at the
        cost of requiring more convex inequalities to represent the geometric mean.


    Attributes
    ----------
    w : tuple of ``Fractions``
        A rational approximation of ``p/sum(p)``.
    approx_error : float
        The error in approximating ``p/sum(p)`` with ``w``, given by
        :math:`\\|p/\\mathbf{1}^T p - w \\|_\\infty`
    """

    def __init__(self, x, p: Optional[List[int]] = None, max_denom: int = 1024) -> None:
        """ Implementation details of geo_mean.

        Attributes
        ----------

        w_dyad : tuple of ``Fractions`` whose denominators are all a power of two
            The dyadic completion of ``w``, which is used internally to form the
            inequalities representing the geometric mean.

        tree : ``dict``
            keyed by dyadic tuples, whose values are Sequences of children.
            The children are also dyadic tuples.
            This represents the graph that needs to be formed to represent the
            weighted geometric mean.

        cone_lb : int
            A known lower bound (which is not always tight) on the number of cones
            needed to represent this geometric mean.

        cone_num_over : int
            The number of cones beyond the lower bound that this geometric mean used.
            If 0, we know that it used the minimum possible number of cones.
            Since cone_lb is not always tight, it may be using the minimum number of cones even if
            cone_num_over is not 0.

        cone_num : int
            The number of second order cones used to form this geometric mean

        """
        Expression = cvxtypes.expression()
        if p is not None and isinstance(p, Expression):
            raise TypeError(SECOND_ARG_SHOULD_NOT_BE_EXPRESSION_ERROR_MESSAGE)
        elif p is not None and hasattr(p, '__getitem__'):
            p = np.array(p)
            idxs = p > 0
            if isinstance(x, list):
                x = np.array(x)
            if x.ndim == 0:
                x = Expression.cast_to_const(promote(x, shape=(1,)))[idxs]
            else:
                x = Expression.cast_to_const(x)[idxs]
            p = p[idxs]
        super(geo_mean, self).__init__(x)

        x = self.args[0]
        if x.is_vector():
            n = 1 if x.ndim == 0 else max(x.shape)
        else:
            raise ValueError('x must be a row or column vector.')

        if p is None:
            p = [1]*n
        self.p = p

        if len(p) != n:
            raise ValueError('x and p must have the same number of elements.')

        if any(v < 0 for v in p) or sum(p) <= 0:
            raise ValueError('powers must be nonnegative and not all zero.')

        self.w, self.w_dyad = fracify(p, max_denom)
        self.approx_error = approx_error(p, self.w)

        self.tree = decompose(self.w_dyad)

        # known lower bound on number of cones needed to represent w_dyad
        self.cone_lb = lower_bound(self.w_dyad)

        # number of cones used past known lower bound
        self.cone_num_over = over_bound(self.w_dyad, self.tree)

        # number of cones used
        self.cone_num = self.cone_lb + self.cone_num_over

    # Returns the (weighted) geometric mean of the elements of x.
    def numeric(self, values) -> float:
        values = np.array(values[0]).flatten()
        val = 1.0
        for x, p in zip(values, self.w):
            val *= x**float(p)
        return val

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        # No special case when only one non-zero weight.
        selection = np.array([w_i > 0 for w_i in self.w])
        return [self.args[0][selection > 0] >= 0]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        x = np.array(values[0])
        # No special case when only one non-zero weight.
        w_arr = np.array([float(w_i) for w_i in self.w])
        # Outside domain.
        if np.any(x[w_arr > 0] <= 0):
            return [None]
        else:
            D = w_arr/x.ravel(order='F')*self.numeric(values)
            return [sp.csc_matrix(D).T]

    def name(self) -> str:
        return "%s(%s, (%s))" % (self.__class__.__name__,
                                 self.args[0].name(),
                                 ', '.join(str(v) for v in self.w))

    def pretty_tree(self) -> None:
        print(prettydict(self.tree))

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
        return (True, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return True

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def get_data(self):
        return [self.w, self.w_dyad, self.tree]

    def copy(self, args=None, id_objects=None):
        """Returns a shallow copy of the geo_mean atom.

        Parameters
        ----------
        args : list, optional
            The arguments to reconstruct the atom. If args=None, use the
            current args of the atom.

        Returns
        -------
        geo_mean atom
        """
        if args is None:
            args = self.args
        # Avoid calling __init__() directly as we do not have p and max_denom.
        copy = type(self).__new__(type(self))
        super(type(self), copy).__init__(*args)
        # Emulate __init__()
        copy.w, copy.w_dyad, copy.tree = self.get_data()
        copy.approx_error = self.approx_error
        copy.cone_lb = self.cone_lb
        copy.cone_num_over = self.cone_num_over
        copy.cone_num = self.cone_num
        return copy
