# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DNLP (Disciplined Nonlinear Programming) is an extension of CVXPY to general nonlinear programming. It allows smooth functions to be freely mixed with nonsmooth convex and concave functions, with rules governing how nonsmooth functions can be used.

For theoretical foundation, see: [Disciplined Nonlinear Programming](https://web.stanford.edu/~boyd/papers/dnlp.html)

## Build and Development Commands

```bash
# Install IPOPT solver (required for NLP - use conda, NOT pip)
conda install -c conda-forge cyipopt

# Install from source (development mode)
pip install -e .

# Install pre-commit hooks
pip install pre-commit && pre-commit install

# Run all tests
pytest cvxpy/tests/

# Run a specific test file
pytest cvxpy/tests/test_dgp.py

# Run a specific test method
pytest cvxpy/tests/test_dgp.py::TestDgp::test_product

# Run NLP-specific tests
pytest cvxpy/tests/nlp_tests/

# Lint with ruff
ruff check cvxpy

# Auto-fix lint issues
ruff check --fix cvxpy

# Build documentation
cd doc && make html
```

## Solving with DNLP

```python
import cvxpy as cp
import numpy as np

x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(objective), constraints)

# Initial point required for NLP solvers
x.value = np.ones(n)

# Solve with nlp=True
prob.solve(nlp=True, solver=cp.IPOPT)

# Optional: Run multiple solves with random initial points, return best
prob.solve(nlp=True, solver=cp.IPOPT, best_of=5)
```

## Supported NLP Solvers

| Solver | License | Installation |
|--------|---------|--------------|
| [IPOPT](https://github.com/coin-or/Ipopt) | EPL-2.0 | `conda install -c conda-forge cyipopt` |
| [Knitro](https://www.artelys.com/solvers/knitro/) | Commercial | `pip install knitro` (requires license) |
| [COPT](https://www.copt.de/) | Commercial | Requires license |
| [Uno](https://github.com/cuter-testing/uno) | Open source | See Uno documentation |

## Code Style

- Uses ruff for linting (configured in `pyproject.toml`)
- Target Python version: 3.11+
- Line length: 100 characters
- **IMPORTANT: IMPORTS AT THE TOP** of files - circular imports are the only exception

## License Header

New files should include the Apache 2.0 license header:
```python
"""
Copyright 2025, the CVXPY developers

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

## Architecture

### Expression System

Expressions form an AST (Abstract Syntax Tree):
- **Expression** (base) → Variable, Parameter, Constant, Atom
- **Atom** subclasses implement mathematical functions (in `cvxpy/atoms/`)
- Each atom defines curvature, sign, and disciplined programming rules

### Problem Types

CVXPY supports multiple disciplined programming paradigms:
- **DCP** (Disciplined Convex Programming) - standard convex problems
- **DGP** (Disciplined Geometric Programming) - geometric programs
- **DQCP** (Disciplined Quasiconvex Programming) - quasiconvex programs
- **DNLP** (Disciplined Nonlinear Programming) - smooth nonlinear programs (this extension)

### Reduction Pipeline

Problems are transformed through a chain of reductions before solving:
```
Problem → [Reductions] → Canonical Form → Solver
```

Key reduction classes in `cvxpy/reductions/`:
- `Reduction` base class with `accepts()`, `apply()`, `invert()` methods
- `Chain` composes multiple reductions
- `SolvingChain` orchestrates the full solve process

For DNLP: `CvxAttr2Constr` → `Dnlp2Smooth` → `NLPSolver`

### Solver Categories

- **ConicSolvers** (`cvxpy/reductions/solvers/conic_solvers/`) - SCS, Clarabel, ECOS, etc.
- **QPSolvers** (`cvxpy/reductions/solvers/qp_solvers/`) - OSQP, ProxQP, etc.
- **NLPSolvers** (`cvxpy/reductions/solvers/nlp_solvers/`) - IPOPT, Knitro, COPT, Uno

### NLP System

The NLP infrastructure provides oracle-based interfaces for nonlinear solvers:
- `nlp_solver.py` - Base `NLPsolver` class with:
  - `Bounds` class: extracts variable/constraint bounds from problem
  - `Oracles` class: provides function and derivative oracles (objective, gradient, constraints, jacobian, hessian)
- `dnlp2smooth.py` - Transforms DNLP problems to smooth form via `Dnlp2Smooth` reduction
- DNLP validation: expressions must be smooth (ESR and HSR)
- Problem validity checked via `problem.is_dnlp()` method

### Diff Engine (SparseDiffPy)

The automatic differentiation engine is provided by the [SparseDiffPy](https://github.com/SparseDifferentiation/SparseDiffPy) package (`pip install sparsediffpy`), which wraps the [SparseDiffEngine](https://github.com/SparseDifferentiation/SparseDiffEngine) C library. It builds expression trees from CVXPY problems and computes derivatives (gradients, Jacobians, Hessians) for NLP solvers. New diff engine atoms require C-level additions in SparseDiffPy.

## Implementing New Atoms

### For DCP Atoms

1. Create atom class in `cvxpy/atoms/` or `cvxpy/atoms/elementwise/`
2. Implement: `shape_from_args()`, `sign_from_args()`, `is_atom_convex()`, `is_atom_concave()`, `is_incr()`, `is_decr()`, `numeric()`
3. Create canonicalizer in `cvxpy/reductions/dcp2cone/canonicalizers/`
4. Register in `canonicalizers/__init__.py` by adding to `CANON_METHODS` dict
5. Export in `cvxpy/atoms/__init__.py`

### For DNLP Support

1. Create a canonicalizer in `cvxpy/reductions/dnlp2smooth/canonicalizers/`
2. The canonicalizer converts non-smooth atoms to smooth equivalents using auxiliary variables
3. Register in `canonicalizers/__init__.py` by adding to `SMOOTH_CANON_METHODS` dict
4. Ensure the atom has proper `is_smooth()`, `is_esr()`, `is_hsr()` methods

### DNLP Rules (ESR/HSR)

- **Smooth**: functions that are both ESR and HSR (analogous to affine in DCP)
- **ESR** (Essentially Smooth Respecting): can be minimized or appear in `<= 0` constraints
- **HSR** (Hierarchically Smooth Respecting): can be maximized or appear in `>= 0` constraints

Use `expr.is_smooth()`, `expr.is_esr()`, `expr.is_hsr()` to check expression properties.

## Testing

Tests should be comprehensive but concise. Use `solver=cp.CLARABEL` for tests that call `problem.solve()`.

```python
from cvxpy.tests.base_test import BaseTest
import cvxpy as cp
import numpy as np

class TestMyFeature(BaseTest):
    def test_basic(self) -> None:
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.OPTIMAL)
```

NLP tests are in `cvxpy/tests/nlp_tests/` with Jacobian and Hessian verification tests.
