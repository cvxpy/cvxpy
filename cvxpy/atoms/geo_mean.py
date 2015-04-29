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


import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.affine.index import index
from cvxpy.constraints import SOC
import numpy as np
from fractions import Fraction
from collections import defaultdict


class geo_mean(Atom):
    """ The (weighted) geometric mean of vector ``x``, with optional powers given by ``p``:

    .. math::

        \\left(x_1^{p_1} \cdots x_n^{p_n} \\right)^{\\frac{1}{\mathbf{1}^Tp}}

    The powers ``p`` can be a ``list``, ``tuple``, or ``numpy.array`` of nonnegative
    ``int``, ``float``, or ``Fraction`` objects with nonzero sum.

    If not specified, ``p`` defaults to a vector of all ones, giving the
    **unweighted** geometric mean

    .. math::

        x_1^{1/n} \cdots x_n^{1/n}.

    The geometric mean includes an implicit constraint that :math:`x_i \geq 0`
    whenever :math:`p_i > 0`. If :math:`p_i = 0`, :math:`x_i` will be unconstrained.

    The only exception to this rule occurs when
    ``p`` has exactly one nonzero element, say, ``p_i``, in which case
    ``geo_mean(x, p)`` is equivalent to ``x_i`` (without the nonnegativity constraint).
    A specific case of this is when :math:`x \in \mathbf{R}^1`.


    .. note::

        Generally, ``p`` cannot be represented exactly, so a rational,
        i.e., fractional, **approximation** must be made.

        Internally, ``geo_mean`` immediately computes an approximate normalized
        weight vector :math:`w \\approx p/\mathbf{1}^Tp`
        and the ``geo_mean`` atom is represented as

        .. math::

            x_1^{w_1} \cdots x_n^{w_n},

        where the elements of ``w`` are ``Fraction`` objects that sum to exactly 1.

        The maximum denominator used in forming the rational approximation
        is given by ``max_denom``, which defaults to 1024, but can be adjusted
        to modify the accuracy of the approximations.

        The approximating ``w`` and the approximation error can be
        found through the attributes ``geo_mean.w`` and ``geo_mean.approx_error``.


    Examples
    --------

    The weights ``w`` can be seen from the string representation of the ``geo_mean`` object, or through
    the ``w`` attribute.

    >>> from cvxpy import Variable, geo_mean, Problem, Maximize
    >>> x = Variable(3, name='x')
    >>> print(geo_mean(x))
    geo_mean(x, (1/3, 1/3, 1/3))
    >>> g = geo_mean(x, [1, 2, 1])
    >>> g.w
    (Fraction(1, 4), Fraction(1, 2), Fraction(1, 4))

    Floating point numbers with few decimal places can sometimes be represented exactly. The approximation
    error between ``w`` and ``p/sum(p)`` is given by the ``approx_error`` attribute.

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

    The weight vector ``p`` can contain combinations of ``int``, ``float``, and ``Fraction`` objects.

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
    x : cvxpy.Variable
        A column or row vector whose elements we will take the geometric mean of.

    p : Sequence (list, tuple, numpy.array, ...) of ``int``, ``float``, or ``Fraction`` objects
        A vector of weights for the weighted geometric mean

        When ``p`` is a sequence of ``int`` and/or ``Fraction`` objects, ``w`` can often be an **exact** representation
        of the weights. An exact representation is sometimes possible when ``p`` has ``float`` elements with only a few
        decimal places.

    max_denom : int
        The maximum denominator to use in approximating ``p/sum(p)`` with ``geo_mean.w``. If ``w`` is not an exact
        representation, increasing ``max_denom`` **may** offer a more accurate representation, at the cost of requiring
        more convex inequalities to represent the geometric mean.


    Attributes
    ----------
    w : tuple of ``Fractions``
        A rational approximation of ``p/sum(p)``.
    approx_error : float
        The error in approximating ``p/sum(p)`` with ``w``, given by :math:`\|p/\mathbf{1}^T p - w \|_\infty`
    """

    two = lu.create_const(2, (1, 1))

    def __init__(self, x, p=None, max_denom=1024):
        """ Implementation details of geo_mean.

        Attributes
        ----------

        w_dyad : tuple of ``Fractions`` whose denominators are all a power of two
            The dyadic completion of ``w``, which is used internally to form the inequalities representing the
            geometric mean.

        tree : ``dict``
            keyed by dyadic tuples, whose values are Sequences of children.
            The children are also dyadic tuples.
            This represents the graph that needs to be formed to represent the weighted geometric mean.

        cone_lb : int
            A known lower bound (which is not always tight) on the number of cones needed to represent this
            geometric mean.

        cone_num_over : int
            The number of cones beyond the lower bound that this geometric mean used.
            If 0, we know that it used the minimum possible number of cones.
            Since cone_lb is not always tight, it may be using the minimum number of cones even if
            cone_num_over is not 0.

        cone_num : int
            The number of second order cones used to form this geometric mean

        """
        super(geo_mean, self).__init__(x)

        if not (isinstance(max_denom, int) and max_denom > 0):
            raise ValueError('max_denom must be a positive integer.')

        x = self.args[0]
        if x.size[0] == 1:
            n = x.size[1]
        elif x.size[1] == 1:
            n = x.size[0]
        else:
            raise ValueError('x must be a row or column vector.')

        if p is None:
            p = [1]*n

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
    @Atom.numpy_numeric
    def numeric(self, values):
        values = np.array(values[0]).flatten()
        val = 1.0
        for x, p in zip(values, self.w):
            val *= x**p
        return val

    def name(self):
        return "%s(%s, (%s))" % (self.__class__.__name__,
                                 self.args[0].name(),
                                 ', '.join(str(v) for v in self.w))

    def pretty_tree(self):
        print(prettydict(self.tree))

    def shape_from_args(self):
        return u.Shape(1, 1)

    def sign_from_args(self):
        return u.Sign.POSITIVE

    def func_curvature(self):
        return u.Curvature.CONCAVE

    def monotonicity(self):
        return [u.monotonicity.INCREASING]

    def validate_arguments(self):
        # since correctly validating arguments with this function is tricky,
        # we do it in __init__ instead.
        pass

    def get_data(self):
        return self.w, self.w_dyad, self.tree

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
        w, w_dyad, tree = data
        d = defaultdict(lambda: lu.create_var((1, 1)))

        constraints = []

        vars_ = [index.get_index(arg_objs[0], [], i, 0) for i in range(len(w))] + [d[w_dyad]]
        for i, (p, v) in enumerate(zip(w_dyad, vars_)):
            if p > 0:
                tmp = [0]*len(w_dyad)
                tmp[i] = 1
                d[tuple(tmp)] = v

        for elem, children in tree.items():
            if 1 not in elem:
                constraints += [gm(d[elem], d[children[0]], d[children[1]])]

        return d[w_dyad], constraints


def gm(w, x, y):
    """ Form the basic second order cone constraint to form the geometric mean expression
        w <= sqrt(x*y)
        SOC(x + y, [y - x, 2*w])
    """
    # w <= sqrt(x*y)
    # SOC(x + y, [y - x, 2*w])
    return SOC(lu.sum_expr([x, y]),
               [lu.sub_expr(x, y),
                lu.mul_expr(geo_mean.two, w, (1, 1))])


# helper functions

def is_power2(num):
    """ Test if num is a nonnegative integer power of 2.

    Examples
    --------
    >>> is_power2(4)
    True
    >>> is_power2(2**10)
    True
    >>> is_power2(1)
    True
    >>> is_power2(1.0)
    False
    >>> is_power2(0)
    False
    >>> is_power2(-4)
    False
    """
    return isinstance(num, int) and num > 0 and not (num & (num - 1))


def is_dyad(frac):
    """ Test if frac is a nonnegative dyadic fraction or integer.

    Examples
    --------
    >>> is_dyad(Fraction(1,4))
    True
    >>> is_dyad(Fraction(1,3))
    False
    >>> is_dyad(0)
    True
    >>> is_dyad(1)
    True
    >>> is_dyad(-Fraction(1,4))
    False
    >>> is_dyad(Fraction(1,6))
    False

    """
    if isinstance(frac, int) and frac >= 0:
        return True
    elif isinstance(frac, Fraction) and frac >= 0 and is_power2(frac.denominator):
        return True
    else:
        return False


def is_dyad_weight(w):
    """ Test if a vector is a valid dyadic weight vector.

        w must be nonnegative, sum to 1, and have integer or dyadic fractional elements.

        Examples
        --------
        >>> is_dyad_weight((Fraction(1,2), Fraction(1,2)))
        True
        >>> is_dyad_weight((Fraction(1,3), Fraction(2,3)))
        False
        >>> is_dyad_weight((0, 1, 0))
        True
    """
    return is_weight(w) and all(is_dyad(f) for f in w)


def is_weight(w):
    """ Test if w is a valid weight vector.
        w must have nonnegative integer or fractional elements, and sum to 1.

    Examples
    --------
    >>> is_weight((Fraction(1,3), Fraction(2,3)))
    True
    >>> is_weight((Fraction(2,3), Fraction(2,3)))
    False
    >>> is_weight([.1, .9])
    False
    >>> import numpy as np
    >>> w = np.array([.1, .9])
    >>> is_weight(w)
    False
    >>> w = np.array([0, 0, 1])
    >>> is_weight(w)
    True
    >>> w = (0,1,0)
    >>> is_weight(w)
    True

    """
    if isinstance(w, np.ndarray):
        w = w.tolist()
    return (all(v >= 0 and isinstance(v, (int, Fraction)) for v in w)
            and sum(w) == 1)


def fracify(a, max_denom=1024, force_dyad=False):
    """ Return a valid fractional weight tuple (and its dyadic completion)
        to represent the weights given by ``a``.

        When the input tuple contains only integers and fractions,
        ``fracify`` will try to represent the weights exactly.

    Parameters
    ----------
    a : Sequence
        Sequence of numbers (ints, floats, or Fractions) to be represented
        with fractional weights.

    max_denom : int
        The maximum denominator allowed for the fractional representation.
        When the fractional representation is not exact, increasing
        ``max_denom`` will typically give a better approximation.

        Note that ``max_denom`` is actually replaced with the largest power
        of 2 >= ``max_denom``.

    force_dyad : bool
        If ``True``, we force w to be a dyadic representation so that ``w == w_dyad``.
        This means that ``w_dyad`` does not need an extra dummy variable.
        In some cases, this may reduce the number of second-order cones needed to
        represent ``w``.

    Returns
    -------
    w : tuple
        Approximation of ``a/sum(a)`` as a tuple of fractions.

    w_dyad : tuple
        The dyadic completion of ``w``.

        That is, if w has fractions with denominators that are not a power of 2,
        and ``len(w) == n`` then w_dyad has length n+1, dyadic fractions for elements,
        and ``w_dyad[:-1]/w_dyad[n] == w``.

        Alternatively, the ratios between the
        first n elements of ``w_dyad`` are equal to the corresponding ratios between
        the n elements of ``w``.

        The dyadic completion of w is needed to represent the weighted geometric
        mean with weights ``w`` as a collection of second-order cones.

        The appended element of ``w_dyad`` is typically a dummy variable.

    Examples
    --------
    >>> w, w_dyad = fracify([1, 2, 3])
    >>> w
    (Fraction(1, 6), Fraction(1, 3), Fraction(1, 2))
    >>> w_dyad
    (Fraction(1, 8), Fraction(1, 4), Fraction(3, 8), Fraction(1, 4))

    >>> w, w_dyad = fracify((1, 1, 1, 1, 1))
    >>> w
    (Fraction(1, 5), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5), Fraction(1, 5))
    >>> w_dyad
    (Fraction(1, 8), Fraction(1, 8), Fraction(1, 8), Fraction(1, 8), Fraction(1, 8), Fraction(3, 8))

    >>> w, w_dyad = fracify([.23, .56, .87])
    >>> w
    (Fraction(23, 166), Fraction(28, 83), Fraction(87, 166))
    >>> w_dyad
    (Fraction(23, 256), Fraction(7, 32), Fraction(87, 256), Fraction(45, 128))

    >>> w, w_dyad = fracify([3, Fraction(1, 2), Fraction(3, 5)])
    >>> w
    (Fraction(30, 41), Fraction(5, 41), Fraction(6, 41))
    >>> w_dyad
    (Fraction(15, 32), Fraction(5, 64), Fraction(3, 32), Fraction(23, 64))

    Can also mix integer, Fraction, and floating point types.

    >>> w, w_dyad = fracify([3.4, 8, Fraction(3, 2)])
    >>> w
    (Fraction(34, 129), Fraction(80, 129), Fraction(5, 43))
    >>> w_dyad
    (Fraction(17, 128), Fraction(5, 16), Fraction(15, 256), Fraction(127, 256))

    Forcing w to be dyadic makes it its own dyadic completion.

    >>> w, w_dyad = fracify([3.4, 8, Fraction(3, 2)], force_dyad=True)
    >>> w
    (Fraction(135, 512), Fraction(635, 1024), Fraction(119, 1024))
    >>> w_dyad
    (Fraction(135, 512), Fraction(635, 1024), Fraction(119, 1024))

    A standard basis unit vector should yield itself.

    >>> w, w_dyad = fracify((0, 0.0, 1.0))
    >>> w
    (Fraction(0, 1), Fraction(0, 1), Fraction(1, 1))
    >>> w_dyad
    (Fraction(0, 1), Fraction(0, 1), Fraction(1, 1))

    A dyadic weight vector should also yield itself.

    >>> a = (Fraction(1,2), Fraction(1,8), Fraction(3,8))
    >>> w, w_dyad = fracify(a)
    >>> a == w == w_dyad
    True

    Be careful when converting floating points to fractions.

    >>> a = (Fraction(.9), Fraction(.1))
    >>> w, w_dyad = fracify(a)
    Traceback (most recent call last):
    ...
    ValueError: Can't reliably represent the input weight vector.
    Try increasing `max_denom` or checking the denominators of your input fractions.

    The error here is because ``Fraction(.9)`` and ``Fraction(.1)``
    evaluate to ``(Fraction(8106479329266893, 9007199254740992)`` and
    ``Fraction(3602879701896397, 36028797018963968))``.

    """
    if any(v < 0 for v in a):
        raise ValueError('Input powers must be nonnegative.')

    if not (isinstance(max_denom, int) and max_denom > 0):
        raise ValueError('Input denominator must be an integer.')

    if isinstance(a, np.ndarray):
        a = a.tolist()

    max_denom = next_pow2(max_denom)
    total = sum(a)

    if force_dyad is True:
        w_frac = make_frac(a, max_denom)
    elif all(isinstance(v, (int, Fraction)) for v in a):
        w_frac = tuple(Fraction(v, total) for v in a)
        d = max(v.denominator for v in w_frac)
        if d > max_denom:
            msg = "Can't reliably represent the input weight vector."
            msg += "\nTry increasing `max_denom` or checking the denominators of your input fractions."
            raise ValueError(msg)
    else:
        # fall through code
        w_frac = tuple(Fraction(float(v)/total).limit_denominator(max_denom) for v in a)
        if sum(w_frac) != 1:
            w_frac = make_frac(a, max_denom)

    return w_frac, dyad_completion(w_frac)


def make_frac(a, denom):
    """ Approximate ``a/sum(a)`` with tuple of fractions with denominator *exactly* ``denom``.

    >>> a = [.123, .345, .532]
    >>> make_frac(a,10)
    (Fraction(1, 10), Fraction(2, 5), Fraction(1, 2))
    >>> make_frac(a,100)
    (Fraction(3, 25), Fraction(7, 20), Fraction(53, 100))
    >>> make_frac(a,1000)
    (Fraction(123, 1000), Fraction(69, 200), Fraction(133, 250))
    """

    a = np.array(a, dtype=float)/sum(a)
    b = [float(v*denom) for v in a]
    b = np.array(b, dtype=int)
    err = b/float(denom) - a

    inds = np.argsort(err)[:(denom - sum(b))]
    b[inds] += 1

    denom = int(denom)
    b = b.tolist()

    return tuple(Fraction(v, denom) for v in b)


def dyad_completion(w):
    """ Return the dyadic completion of ``w``.

        Return ``w`` if ``w`` is already dyadic.

        We assume the input is a tuple of nonnegative Fractions or integers which sum to 1.

    Examples
    --------
    >>> w = (Fraction(1,3), Fraction(1,3), Fraction(1, 3))
    >>> dyad_completion(w)
    (Fraction(1, 4), Fraction(1, 4), Fraction(1, 4), Fraction(1, 4))

    >>> w = (Fraction(1,3), Fraction(1,5), Fraction(7, 15))
    >>> dyad_completion(w)
    (Fraction(5, 16), Fraction(3, 16), Fraction(7, 16), Fraction(1, 16))

    >>> w = (1, 0, 0.0, Fraction(0,1))
    >>> dyad_completion(w)
    (Fraction(1, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1))
    """
    w = tuple(Fraction(v) for v in w)
    d = max(v.denominator for v in w)

    # if extra_index:
    p = next_pow2(d)
    if p == d:
        # the tuple of fractions is already dyadic
        return w
    else:
        # need to add the dummy variable to represent as dyadic
        return tuple(Fraction(v*d, p) for v in w) + (Fraction(p-d, p),)


def approx_error(a_orig, w_approx):
    """ Return the :math:`\ell_\infty` norm error from approximating the vector a_orig/sum(a_orig)
        with the weight vector w_approx.

        That is, return

        .. math:: \|a/\mathbf{1}^T a - w_{\mbox{approx}} \|_\infty


        >>> e = approx_error([1, 1, 1], [Fraction(1,3), Fraction(1,3), Fraction(1,3)])
        >>> e <= 1e-10
        True
    """
    assert all(v >= 0 for v in a_orig)
    assert is_weight(w_approx)
    assert len(a_orig) == len(w_approx)

    w_orig = np.array(a_orig, dtype=float)/sum(a_orig)
    return max(abs(v1-v2) for v1, v2 in zip(w_orig, w_approx))


def next_pow2(n):
    """ Return first power of 2 >= n.

    >>> next_pow2(3)
    4
    >>> next_pow2(8)
    8
    >>> next_pow2(0)
    1
    >>> next_pow2(1)
    1
    """
    p = 2**(len(bin(n))-2)
    if p//2 >= n:
        p //= 2
    return p


def check_dyad(w, w_dyad):
    """Check that w_dyad is a valid dyadic completion of w.

    Parameters
    ----------
    w : Sequence
        Tuple of nonnegative fractional or integer weights that sum to 1.
    w_dyad : Sequence
        Proposed dyadic completion of w.

    Returns
    -------
    bool
        True if w_dyad is a valid dyadic completion of w.


    Examples
    --------
    >>> w = (Fraction(1,3), Fraction(1,3), Fraction(1,3))
    >>> w_dyad =(Fraction(1,4), Fraction(1,4), Fraction(1,4), Fraction(1,4))
    >>> check_dyad(w, w_dyad)
    True

    If the weight vector is already dyadic, it is its own completion.

    >>> w = (Fraction(1,4), 0, Fraction(3,4))
    >>> check_dyad(w, w)
    True

    Integer input should also be accepted

    >>> w = (1, 0, 0)
    >>> check_dyad(w, w)
    True

    w is not a valid weight vector here because it doesn't sum to 1

    >>> w = (Fraction(2,3), 1)
    >>> check_dyad(w, w)
    False

    w_dyad isn't the correct dyadic completion.

    >>> w = (Fraction(2,5), Fraction(3,5))
    >>> w_dyad = (Fraction(3,8), Fraction(4,8), Fraction(1,8))
    >>> check_dyad(w, w_dyad)
    False

    The correct dyadic completion.

    >>> w = (Fraction(2,5), Fraction(3,5))
    >>> w_dyad = (Fraction(2,8), Fraction(3,8), Fraction(3,8))
    >>> check_dyad(w, w_dyad)
    True

    """
    if not (is_weight(w) and is_dyad_weight(w_dyad)):
        return False
    if w == w_dyad:
        # w is its own dyadic completion
        return True
    if len(w_dyad) == len(w) + 1:
        return w == tuple(Fraction(v, 1-w_dyad[-1]) for v in w_dyad[:-1])
    else:
        return False


def split(w_dyad):
    """ Split a tuple of dyadic rationals into two children
    so that d_tup = 1/2*(child1 + child2).

    Here, d_tup, child1, and child2 have nonnegative dyadic rational elements,
    and each tuple sums to 1.

    Basis vectors such as d_tup = (0, 1, 0) will return no children, since they cannot be split.
    """

    # since this should only be called by decompose, assume w_dyad is a valid dyadic weight vector.

    if 1 in w_dyad:
        # then the vector is all zeros with a single 1. can't be split. has no children.
        return ()

    bit = Fraction(1, 1)
    child1 = [Fraction(0)]*len(w_dyad)
    child2 = list(2*f for f in w_dyad)  # assign twice the parent's value to child 2

    while True:
        for ind, val in enumerate(child2):
            if val >= bit:
                child2[ind] -= bit
                child1[ind] += bit
            if sum(child1) == 1:
                return tuple(child1), tuple(child2)
        bit /= 2

    raise ValueError('Something wrong with input {}'.format(w_dyad))


def decompose(w_dyad):
    """ Recursively split dyadic tuples to produce a DAG. A node
    can have multiple parents. Interior nodes in the DAG represent second-order cones
    which must be formed to represent the corresponding weighted geometric mean.

    Return a dictionary keyed by dyadic tuples. The values are a list of that tuple's children.
    The dictionary will allow us to re-use nodes whose tuple we have already seen, which
    reduces the number of second-order cones that need to be formed.
    We use an OrderedDict so that the root node is the first element of tree.keys().
    """

    if not is_dyad_weight(w_dyad):
        return ValueError('input must be a dyadic weight vector.')
    tree = {}
    todo = [tuple(w_dyad)]
    for t in todo:
        if t not in tree:
            tree[t] = split(t)
            todo += list(tree[t])

    return tree


def prettytuple(t):
    """ Use the string representation of objects in a tuple.
    """
    return '(' + ', '.join(str(f) for f in t) + ')'


def get_max_denom(tup):
    """ Get the maximum denominator in a sequence of ``Fraction`` and ``int`` objects
    """
    return max(Fraction(f).denominator for f in tup)


def prettydict(d):
    """ Print keys of a dictionary with children (expected to be a Sequence) indented underneath.

    Used for printing out trees of second order cones to represent weighted geometric means.

    """
    keys = sorted(list(d.keys()), key=get_max_denom, reverse=True)
    result = ""
    for tup in keys:
        children = sorted(d[tup], key=get_max_denom, reverse=False)
        result += prettytuple(tup) + '\n'
        for child in children:
            result += '  ' + prettytuple(child) + '\n'

    return result


def lower_bound(w_dyad):
    """ Return a lower bound on the number of cones needed to represent the tuple.
        Based on two simple lower bounds.

    Examples
    --------
    >>> lower_bound((0,1,0))
    0
    >>> lower_bound((Fraction(1, 2), Fraction(1,2)))
    1
    >>> lower_bound((Fraction(1, 4), Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)))
    3
    >>> lower_bound((Fraction(1,8), Fraction(7,8)))
    3
    """
    assert is_dyad_weight(w_dyad)
    md = get_max_denom(w_dyad)

    lb1 = len(bin(md))-3

    # don't include zero entries
    lb2 = sum(1 if e != 0 else 0 for e in w_dyad) - 1
    return max(lb1, lb2)


def over_bound(w_dyad, tree):
    """ Return the number of cones in the tree beyond the known lower bounds.
        if it is zero, then we know the tuple can't be represented in fewer cones.
    """
    nonzeros = sum(1 for e in w_dyad if e != 0)
    return len(tree) - lower_bound(w_dyad) - nonzeros
