Atoms
====================
Atoms are mathematical functions that can be applied to
:class:`~cvxpy.expressions.expression.Expression` instances.
Applying an atom to an expression yields another expression.
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

Atom
----

.. autoclass:: cvxpy.atoms.atom.Atom
    :members: is_atom_convex, is_atom_concave, is_atom_affine,
              is_atom_log_log_convex, is_atom_log_log_concave,
              is_atom_log_log_affine, is_incr, is_decr, grad, domain
    :undoc-members:
    :show-inheritance:
