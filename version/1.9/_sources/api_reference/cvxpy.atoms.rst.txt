Atoms
====================

An atom (with a lower-case "a") is a mathematical function that can be applied to
:class:`~cvxpy.Expression` objects and returns an :class:`~cvxpy.Expression` object.

Atoms and compositions thereof are precisely the mechanisms that allow you to
build up mathematical expression trees in CVXPY.

Every atom is tagged with information about its domain, sign, curvature,
log-log curvature, and monotonicity; this information lets atom instances
reason about whether or not they are DCP or DGP. See the :ref:`functions` page
for a compact, accessible summary of each atom's attributes.


.. toctree::
    Affine Atoms <cvxpy.atoms.affine>
    Elementwise Atoms <cvxpy.atoms.elementwise>
    Other Atoms <cvxpy.atoms.other_atoms>

Representation of atoms
-----------------------

From an implementation perspective, an atom might be the constructor for some class.
For example, the atom :math:`X \mapsto \lambda_{\max}(X)` is applied by constructing
an instance of the :class:`~cvxpy.atoms.lambda_max.lambda_max` class, which inherits
directly from :class:`~cvxpy.atoms.atom.Atom` and indirectly from :class:`~cvxpy.expressions.expression.Expression`.
Most atoms are implemented this way.

Alternatively, an atom be a wrapper that initializes and returns
an Atom of some other class. For example, running

.. code:: python

	import cvxpy as cp
	X = cp.Variable(shape=(2,2), symmetric=True)
	expr = cp.lambda_min(X)
	print(type(expr))

shows

.. parsed-literal::

	<class 'cvxpy.atoms.affine.unary_operators.NegExpression'>

This happens because *(1)* CVXPY implements :func:`~cvxpy.atoms.lambda_min.lambda_min` as

.. math::

	\lambda_{\min}(X) = -\lambda_{\max}(-X),

*(2)* the negation operator is a class-based atom, and *(3)* the precise type of an Expression is based
on the last class-based atom applied to it (if any such atom has been applied).


Atom
----

.. autoclass:: cvxpy.atoms.atom.Atom
    :members: is_atom_convex, is_atom_concave, is_atom_affine,
              is_atom_log_log_convex, is_atom_log_log_concave,
              is_atom_log_log_affine, is_incr, is_decr, grad, domain
    :undoc-members:
    :show-inheritance:
