# CVXPY Development Guide

## Quick Reference

### Commands
```bash
# Run all tests
pytest cvxpy/tests/

# Run specific test
pytest cvxpy/tests/test_atoms.py::TestAtoms::test_norm_inf

# Lint check
ruff check cvxpy

# Lint with auto-fix
ruff check --fix cvxpy

# Build from source
python setup.py develop
```

## Code Style

### Ruff Configuration
- **Line length**: 100 characters
- **Target Python**: 3.11+
- **Checks**: E, F, I (isort), NPY201, W605
- Pre-commit available: `pip install pre-commit && pre-commit install`

### Critical Rules
- **IMPORTS AT THE TOP** of files
- Add Apache 2.0 license header to all new files

### License Header
```python
"""
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
"""
```

### Common Imports
```python
import cvxpy.settings as s          # Solver/status constants
import cvxpy.utilities as u          # General utilities
import cvxpy.lin_ops.lin_utils as lu # Linear op utilities
from cvxpy.utilities import performance_utils as perf
```

## Project Structure

```
cvxpy/
├── atoms/              # Mathematical functions (exp, log, norm, etc.)
│   ├── affine/         # Shape-preserving ops (reshape, sum, trace)
│   └── elementwise/    # Element-wise ops (exp, log, abs, sqrt)
├── constraints/        # Constraint types (Zero, NonNeg, SOC, PSD)
├── expressions/        # Variable, Parameter, Constant, Expression
├── problems/           # Problem class and Minimize/Maximize
├── reductions/         # Problem transformations
│   ├── dcp2cone/       # DCP → conic canonicalizers
│   ├── dgp2dcp/        # DGP → DCP transforms
│   └── solvers/        # Solver interfaces
│       ├── conic_solvers/
│       └── qp_solvers/
├── lin_ops/            # Linear operator representation
│   └── backends/       # Canonicalization backends
├── utilities/          # Helpers, performance utils
└── tests/              # Unit tests (use pytest to run)
```

## Architecture

### Expression Hierarchy
```
Expression (base)
├── Leaf (terminal nodes)
│   ├── Variable
│   ├── Parameter
│   └── Constant
└── Atom (function applications)
    ├── AffineAtom
    ├── Elementwise
    └── AxisAtom
```

### Reduction Chain
Problems are transformed through a chain of reductions:
```
Problem → Dcp2Cone → ConeMatrixStuffing → Solver
```

Each reduction implements:
- `accepts(problem) → bool` - Can handle this problem?
- `apply(problem) → (new_problem, inverse_data)` - Transform
- `invert(solution, inverse_data) → solution` - Map back

### DCP Rules
Atoms define curvature via:
- `is_atom_convex()` / `is_atom_concave()` - Intrinsic curvature
- `is_incr(idx)` / `is_decr(idx)` - Monotonicity per argument

## Implementing New Atoms

### 1. Create Atom Class
Location: `cvxpy/atoms/` or `cvxpy/atoms/elementwise/`

```python
from typing import Tuple
from cvxpy.atoms.atom import Atom

class my_atom(Atom):
    def __init__(self, x) -> None:
        super().__init__(x)

    def shape_from_args(self) -> Tuple[int, ...]:
        return self.args[0].shape

    def sign_from_args(self) -> Tuple[bool, bool]:
        return (False, False)  # (is_nonneg, is_nonpos)

    def is_atom_convex(self) -> bool:
        return True

    def is_atom_concave(self) -> bool:
        return False

    def is_incr(self, idx: int) -> bool:
        return True

    def is_decr(self, idx: int) -> bool:
        return False

    def numeric(self, values):
        return np.my_function(values[0])
```

### 2. Create Canonicalizer
Location: `cvxpy/reductions/dcp2cone/canonicalizers/`

```python
def my_atom_canon(expr, args):
    from cvxpy.expressions.variable import Variable
    t = Variable(expr.shape)
    constraints = [...]  # Conic constraints enforcing t == my_atom(args[0])
    return t, constraints
```

### 3. Register
In `cvxpy/reductions/dcp2cone/canonicalizers/__init__.py`:
```python
from cvxpy.atoms import my_atom
CANON_METHODS[my_atom] = my_atom_canon
```

### 4. Export
In `cvxpy/atoms/__init__.py`:
```python
from cvxpy.atoms.my_atom import my_atom
```

## Testing

Tests should be **comprehensive but concise and focused**. Cover edge cases without unnecessary verbosity.

### Base Test Pattern
```python
from cvxpy.tests.base_test import BaseTest
import cvxpy as cp
import numpy as np

class TestMyFeature(BaseTest):
    def test_basic(self) -> None:
        x = cp.Variable(2)
        atom = cp.my_atom(x)

        # Test DCP
        self.assertTrue(atom.is_convex())

        # Test numeric
        x.value = np.array([1.0, 2.0])
        self.assertItemsAlmostEqual(atom.value, expected)

    def test_solve(self) -> None:
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        prob.solve()
        self.assertEqual(prob.status, cp.OPTIMAL)
```

### Assertion Helpers
- `self.assertItemsAlmostEqual(a, b, places=5)` - Compare arrays
- `self.assertAlmostEqual(a, b, places=5)` - Compare scalars

## Performance Utilities

```python
from cvxpy.utilities import performance_utils as perf

class MyClass:
    @perf.lazyprop
    def expensive_prop(self):
        """Computed once, cached."""
        return compute()
```

## Constants Reference

### Solvers (`cvxpy/settings.py`)
```python
CLARABEL, CVXOPT, ECOS, GLOP, GUROBI, HIGHS, MOSEK, OSQP,
PIQP, PROXQP, SCS, SCIPY, XPRESS, ...
```

### Status
```python
OPTIMAL, OPTIMAL_INACCURATE, INFEASIBLE, UNBOUNDED,
INFEASIBLE_OR_UNBOUNDED, USER_LIMIT, SOLVER_ERROR
```

## Canon Backend Architecture

Backends in `cvxpy/lin_ops/backends/`:

```
backends/
├── __init__.py      # Re-exports, get_backend() factory
├── base.py          # CanonBackend, TensorRepresentation, TensorView
├── scipy_backend.py # SciPyCanonBackend - stacked sparse (default)
├── coo_backend.py   # CooCanonBackend - 3D COO for large DPP
└── rust_backend.py  # RustCanonBackend - future Rust impl
```

### Backend Selection
```bash
CVXPY_DEFAULT_CANON_BACKEND=SCIPY  # or COO
```

### Key Classes
- `TensorRepresentation`: Sparse 3D COO (data, row, col, parameter_offset)
- `CanonBackend`: Abstract base
- `PythonCanonBackend`: Python implementation with linop methods
- `TensorView`: Tensor operation abstraction

### Backend Tests
- `cvxpy/tests/test_python_backends.py` - Comprehensive tests
- `cvxpy/tests/test_backend_linops.py` - Cross-backend consistency

## Common Mistakes

1. Forgetting to register canonicalizers in `__init__.py`
2. Missing license headers on new files
3. Not running `ruff check` before committing
4. Imports not at top of file
5. Missing `is_incr`/`is_decr` in atoms (breaks DCP)
6. Not testing with Parameters (DPP compliance)
7. Forgetting to export atoms in `cvxpy/atoms/__init__.py`
