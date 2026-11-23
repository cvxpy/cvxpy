<!--
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Expressions and atoms

An [`Expression`](../cvxpy/expressions/expression.py) is a node in an
expression tree. Variables, parameters, and constants are
[`Leaf`](../cvxpy/expressions/leaf.py) nodes.
An [`Atom`](../cvxpy/atoms/atom.py) applies a function to its argument
expressions. Reuse a base class when it matches the operation, such as
[`AffAtom`](../cvxpy/atoms/affine/affine_atom.py) for affine functions,
[`Elementwise`](../cvxpy/atoms/elementwise/elementwise.py) for elementwise
functions, or [`AxisAtom`](../cvxpy/atoms/axis_atom.py) for reductions over
axes.

## Shape, value, and curvature

An atom's `shape_from_args` determines the shape of its output.
Its `numeric` method evaluates the function given argument values.
To check that behavior, assign values to variables or parameters and read
the expression's `.value`. Use constants when constant evaluation is the path
under test.

DCP analysis combines an atom's intrinsic curvature with the curvature of
its arguments. The atom supplies `is_atom_convex` and `is_atom_concave` for
intrinsic curvature and `is_incr(idx)` and `is_decr(idx)` for monotonicity
in each argument. Monotonicity can depend on the arguments' signs.
Test the compositions whose validity depends on the rule you changed.

[DCP tutorial](../doc/source/tutorial/dcp/index.rst) ·
[Atom tests](../cvxpy/tests/test_atoms.py) ·
[Constant atom tests](../cvxpy/tests/test_constant_atoms.py)

## Adding an atom

1. Use an atom with similar shape and curvature rules as a starting point.
   Implement argument validation, shape, sign, curvature, monotonicity,
   and numeric evaluation as appropriate. Check the domain and gradient
   methods of the base class as well.
2. Implement the representation used by the relevant reduction.
   Affine atoms generally supply `graph_implementation` to build a
   linear operator. For a new conic canonicalizer, follow the
   `(expr, args, solver_context=None)` signature and return the replacement
   expression and auxiliary constraints. Register it in `CANON_METHODS`.
3. Export the public symbol and update the function table and API docs.
4. Exercise the public symbol in tests. If the rule depends on how an
   argument enters canonicalization, check the relevant variable and
   parameter cases. See [Testing](testing.md) for how to choose the cases.

[Example canonicalizer](../cvxpy/reductions/dcp2cone/canonicalizers/huber_canon.py) ·
[Canonicalizer registry](../cvxpy/reductions/dcp2cone/canonicalizers/__init__.py) ·
[Atom exports](../cvxpy/atoms/__init__.py) ·
[Function table](../doc/source/functions/functions_table.rst) ·
[Atom API reference](../doc/source/api_reference/cvxpy.atoms.rst)
