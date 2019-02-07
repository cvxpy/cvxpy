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

from fractions import Fraction
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
import numpy as np
from collections import defaultdict
import numbers


def gm(t, x, y):
    length = t.size
    return SOC(t=reshape(x+y, (length,)),
               X=vstack([reshape(x-y, (1, length)), reshape(2*t, (1, length))]),
               axis=0)


def gm_constrs(t, x_list, p):
    """ Form the internal CXVPY constraints to form the weighted geometric mean t <= x^p.

    t <= x[0]^p[0] * x[1]^p[1] * ... * x[n]^p[n]

    where x and t can either be scalar or matrix variables.

    Parameters
    ----------

    t : cvx.Variable
        The epigraph variable

    x_list : list of cvx.Variable objects
        The vector of input variables. Must be the same length as ``p``.

    p : list or tuple of ``int`` and ``Fraction`` objects
        The powers vector. powers must be nonnegative and sum to *exactly* 1.
        Must be the same length as ``x``.

    Returns
    -------
    constr : list
        list of constraints involving elements of x (and possibly t) to form the geometric mean.

    """
    assert is_weight(p)
    w = dyad_completion(p)

    tree = decompose(w)
    d = defaultdict(lambda: Variable(t.shape))
    d[w] = t

    if len(x_list) < len(w):
        x_list += [t]

    assert len(x_list) == len(w)

    for i, (p, v) in enumerate(zip(w, x_list)):
        if p > 0:
            tmp = [0]*len(w)
            tmp[i] = 1
            d[tuple(tmp)] = v

    constraints = []
    for elem, children in tree.items():
        if 1 not in elem:
            constraints += [gm(d[elem], d[children[0]], d[children[1]])]

    return constraints


def pow_high(p, max_denom=1024):
    """ Return (t,1,x) power tuple

        x <= t^(1/p) 1^(1-1/p)

        user wants the epigraph variable t
    """
    assert p > 1
    p = Fraction(1/Fraction(p)).limit_denominator(max_denom)
    if 1/p == int(1/p):
        return int(1/p), (p, 1-p)
    return 1/p, (p, 1-p)


def pow_mid(p, max_denom=1024):
    """ Return (x,1,t) power tuple

        t <= x^p 1^(1-p)

        user wants the epigraph variable t
    """
    assert 0 < p < 1
    p = Fraction(p).limit_denominator(max_denom)
    return p, (p, 1-p)


def pow_neg(p, max_denom=1024):
    """ Return (x,t,1) power tuple

        1 <= x^(p/(p-1)) t^(-1/(p-1))

        user wants the epigraph variable t
    """
    assert p < 0
    p = Fraction(p)
    p = Fraction(p/(p-1)).limit_denominator(max_denom)
    return p/(p-1), (p, 1-p)


def is_power2(num):
    """ Test if num is a positive integer power of 2.

    .. note::
        Fails if num is a np.integer type like np.int32, np.int64, etc.
        This seems to be a Python 3 issue.
        Make sure to convert all integers to the native python ``int`` type.

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
    return isinstance(num, numbers.Integral) and num > 0 and not (num & (num - 1))


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
    if isinstance(frac, numbers.Integral) and frac >= 0:
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
    valid_elems = all(v >= 0 and
                      isinstance(v, (numbers.Integral, Fraction)) for v in w)
    return valid_elems and sum(w) == 1


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

    if not (isinstance(max_denom, numbers.Integral) and max_denom > 0):
        raise ValueError('Input denominator must be an integer.')

    if isinstance(a, np.ndarray):
        a = a.tolist()

    max_denom = next_pow2(max_denom)
    total = sum(a)

    if force_dyad is True:
        w_frac = make_frac(a, max_denom)
    elif all(isinstance(v, (numbers.Integral, Fraction)) for v in a):
        w_frac = tuple(Fraction(v, total) for v in a)
        d = max(v.denominator for v in w_frac)
        if d > max_denom:
            msg = ("Can't reliably represent the input weight vector."
                   "\nTry increasing `max_denom` or checking the denominators "
                   "of your input fractions.")
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

    denom = np.int32(denom)
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
    """ Return the :math:`\\ell_\\infty` norm error from approximating the vector a_orig/sum(a_orig)
        with the weight vector w_approx.

        That is, return

        .. math:: \\|a/\\mathbf{1}^T a - w_{\\mbox{approx}} \\|_\\infty


        >>> e = approx_error([1, 1, 1], [Fraction(1,3), Fraction(1,3), Fraction(1,3)])
        >>> e <= 1e-10
        True
    """
    assert all(v >= 0 for v in a_orig)
    assert is_weight(w_approx)
    assert len(a_orig) == len(w_approx)

    w_orig = np.array(a_orig, dtype=float)/sum(a_orig)
    return float(max(abs(v1-v2) for v1, v2 in zip(w_orig, w_approx)))


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
    if n <= 0:
        return 1

    n2 = 1 << int(n).bit_length()
    if n2 >> 1 == n:
        return n
    else:
        return n2


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
        raise ValueError('input must be a dyadic weight vector. got: {}'.format(w_dyad))

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
