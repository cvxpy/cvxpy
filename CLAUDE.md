# CVXPY Development Guide

## Quick Reference

### Commands
```bash
# Install in development mode
pip install -e .

# Install pre-commit hooks (required)
pip install pre-commit && pre-commit install

# Run all tests
pytest cvxpy/tests/

# Run specific test
pytest cvxpy/tests/test_atoms.py::TestAtoms::test_norm_inf
```

## Code Style

- **Line length**: 100 characters
- **IMPORTANT: IMPORTS AT THE TOP** of files - circular imports are the only exception
- **IMPORTANT:** Add Apache 2.0 license header to all new files

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

### DGP (Disciplined Geometric Programming)
DGP problems use log-log curvature instead of standard curvature. Transformed to DCP via `dgp2dcp` reduction.

### DQCP (Disciplined Quasiconvex Programming)
DQCP extends DCP to quasiconvex functions. Solved via bisection on a parameter. Transformed via `dqcp2dcp` reduction.

### DPP (Disciplined Parametrized Programming)
DPP enables efficient re-solving when only `Parameter` values change. CVXPY caches the canonicalization and reuses it.

**How it works**: Parameters are treated as affine (not constant) for curvature analysis. This means:
- `param * param` → NOT DPP (quadratic in params)
- `param * variable` → DPP (affine in params, params only in one factor)
- `cp.norm(param)` in constraint → NOT DPP (nonlinear in params)

Check with `problem.is_dpp()`. See `cvxpy/utilities/scopes.py` for implementation.

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
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.solver_context import SolverInfo

def my_atom_canon(expr, args, solver_context: SolverInfo | None = None):
    x = args[0]
    t = Variable(expr.shape)
    # For CONVEX atoms: use t >= f(x)
    #   When minimizing, optimizer pushes t down to equality: t = f(x)
    # For CONCAVE atoms: use t <= f(x)
    #   When maximizing, optimizer pushes t up to equality: t = f(x)
    constraints = [t >= x]  # Example for convex atom
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

**IMPORTANT:** Use `solver=cp.CLARABEL` for tests that call `problem.solve()` - it's the default open-source solver.

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
        prob.solve(solver=cp.CLARABEL)
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

## Canon Backend Architecture (Advanced)

Most contributors don't need to modify backends. This section is for performance optimization work.

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

## Common Mistakes to Avoid

1. **Forgetting to register canonicalizers** in `canonicalizers/__init__.py`
2. **Forgetting to export atoms** in `cvxpy/atoms/__init__.py`
3. Missing `is_incr`/`is_decr` methods in atoms (breaks DCP analysis)
4. Not testing with `Parameter` objects (DPP compliance)
5. Missing license headers on new files
