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

# Testing

## Choose cases that can fail for different reasons

Start with a small, deterministic fixture and an expected result that a
reviewer can compute independently. Give each additional case a purpose.
Parameterize when the cases cover different branches, backends, shape rules,
or boundaries that the change can affect. Use descriptive case IDs when the
inputs alone leave the distinction unclear.

Choose dimensions and entries that make a wrong axis or permutation visible.
If a property requires square matrices, keep the matrix axes square and
use different sizes for the batch axes when possible. If a fix concerns one
backend, one explicit check of that backend can cover the regression.

## Test at the layer that owns the behavior

- For expression behavior, assert a value, shape, property, or gradient
  directly. The batched trace test assigns values to two broadcastable
  variables, checks the two batch dimensions of the output, and compares its
  value with a NumPy reference. [`test_trace_batched_product`](../cvxpy/tests/test_atoms.py)
- For a canonicalization change, inspect the reduction or compile with
  `get_problem_data()`. The solver argument selects the data format.
  Check the property the fix needs to preserve, such as the coefficient
  action on a known vector or the expected cone dimensions.
- Solve when the test needs an optimizer, dual variables, solver status,
  or other behavior that occurs during a solve. Explain that dependency
  in the docstring. Use Clarabel when the test needs a conic solver and
  the behavior is independent of the solver choice. Use the solver under
  test for solver-specific behavior.

## Describe the test in a docstring

Give each new or substantially revised test a natural-language docstring.
Describe the setup, the operation being tested, and how the test decides
whether the result is correct. One sentence can suffice for a simple check.
For a less obvious case, explain why its shape, values, or parameter
choices matter. If an assertion depends on a representation detail such
as the order of auxiliary variables, state which assumption to revisit
if an equivalent reformulation makes the test fail.

The following example compiles a sparse equality using SciPy.
The problem has one variable, stored in Fortran order, and the equality
fixes every entry. The expected vector must satisfy `A @ expected == b`.
This assertion allows equivalent changes to the signs or ordering of
the equality rows. The three-dimensional sparse fixture requires SciPy
1.15 or newer. [SciPy release notes](https://docs.scipy.org/doc/scipy/release/1.15.0-notes.html#scipy-sparse-improvements)

```python
import cvxpy as cp
import numpy as np
import scipy.sparse as sp


def test_sparse_constant_equality():
    """Compile an equality fixing a variable to a sparse three-dimensional array.

    Place a single nonzero at (1, 0, 1) so the third coordinate affects its
    flattened position. Assert that the known value satisfies the compiled equality.
    """
    value = sp.coo_array(([7.0], ([1], [0], [1])), shape=(2, 3, 4))
    x = cp.Variable(value.shape)
    problem = cp.Problem(cp.Minimize(0), [x == value])
    data, _, _ = problem.get_problem_data(cp.CLARABEL, canon_backend=cp.SCIPY_CANON_BACKEND)

    expected = value.toarray().ravel(order="F")
    assert data["dims"].zero == expected.size
    np.testing.assert_allclose(data["A"] @ expected, data["b"])
```

The [COO sparse-constant regression](https://github.com/cvxpy/cvxpy/pull/3625)
uses this check with the COO backend to detect an incorrectly flattened
coordinate.

## Run the tests

Use the neighboring tests as a guide to assertion helpers and fixtures.
Many tests inherit from [`BaseTest`](../cvxpy/tests/base_test.py), which
provides `assertItemsAlmostEqual` for arrays in addition to the usual
`unittest` assertions.

Start with the test or file for the changed behavior. For a regression,
run the test with the fix absent and confirm it fails for the expected
reason. Then run it with the fix and run the surrounding tests. Use the
full suite when the change can affect unrelated atoms, reductions, or
solvers.

```bash
pytest cvxpy/tests/test_atoms.py::TestAtoms::test_norm_inf
pytest cvxpy/tests/test_atoms.py cvxpy/tests/test_constant_atoms.py
pytest cvxpy/tests/test_canon_methods.py
pytest -rs cvxpy/tests/test_conic_solvers.py -k clarabel
pytest cvxpy/tests/
```

Optional-solver tests may skip when the solver or its license is
unavailable. Use `pytest -rs` to check which tests actually ran.
Run tests marked `knitro` in a separate process from other native solver
tests. The NLP workflow uses these two invocations.

```bash
pytest -m "not knitro" cvxpy/tests/nlp_tests/
pytest -m knitro cvxpy/tests/nlp_tests/
```

Run tests marked `ortools` in a separate process from HiGHS tests.
OR-Tools bundles its own HiGHS library.

[Installed solver detection](../cvxpy/reductions/solvers/defines.py) ·
[Pytest markers](../pyproject.toml) ·
[NLP solver workflow](../.github/workflows/test_nlp_solvers.yml) ·
[Optional solver workflow](../.github/workflows/test_optional_solvers.yml)
