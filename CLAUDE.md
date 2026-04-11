# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is a **fork of CVXPY** that adds **Disciplined Nonlinear Programming (DNLP)** support, enabling non-convex optimization with automatic differentiation via SparseDiffPy.

## Commands

```bash
# Install in development mode
pip install -e .

# Install pre-commit hooks (required)
pip install pre-commit && pre-commit install

# Lint
ruff check cvxpy

# Run all tests
pytest cvxpy/tests/

# Run NLP-specific tests
pytest cvxpy/tests/nlp_tests/

# Run a specific test
pytest cvxpy/tests/test_atoms.py::TestAtoms::test_norm_inf
```

## Code Style

- **Line length**: 100 characters
- **Linter**: `ruff`
- **IMPORTANT: IMPORTS AT THE TOP** of files - circular imports are the only exception
- **IMPORTANT:** Add Apache 2.0 license header to all new files:

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

## Architecture

### Expression Hierarchy
```
Expression (base)
├── Leaf (terminal nodes: Variable, Parameter, Constant)
└── Atom (function applications: AffineAtom, Elementwise, AxisAtom)
```

### Reduction Chains

**Standard (DCP) chain:**
```
Problem → [Dgp2Dcp] → [FlipObjective] → Dcp2Cone → CvxAttr2Constr → ConeMatrixStuffing → Solver
```

**NLP chain** (when `nlp=True`):
```
Problem → [FlipObjective] → CvxAttr2Constr → Dnlp2Smooth → NLPSolver (IPOPT/KNITRO/UNO/COPT)
```

Each reduction implements `accepts()`, `apply()`, and `invert()`.
See `cvxpy/reductions/solvers/solving_chain.py` for DCP chain construction.
See `cvxpy/reductions/solvers/nlp_solving_chain.py` for NLP chain construction.

### DCP Rules
Atoms define curvature via `is_atom_convex()`, `is_atom_concave()`, `is_incr(idx)`, `is_decr(idx)`.

### DPP (Disciplined Parametrized Programming)
Parameters are treated as affine (not constant) for curvature analysis. `param * param` is NOT DPP; `param * variable` is DPP. Check with `problem.is_dpp()`. See `cvxpy/utilities/scopes.py`.

## DNLP Architecture

### Core flow (`nlp_solving_chain.py`)
`solve_nlp()` orchestrates DNLP solving:
1. Builds chain: [FlipObjective] → CvxAttr2Constr → Dnlp2Smooth → NLPSolver
2. Sets initial points (from variable values or bounds-based sampling)
3. Supports best-of-N solving with random restarts via `sample_bounds`
4. Caches solver state in `_solver_cache['NLP']` for parametric re-solving

### DNLP expression properties
- `is_dnlp()` on Problem: validates all constraints/objective satisfy DNLP rules
- `is_smooth()`: expression is both linearizable_convex and linearizable_concave
- `is_linearizable_convex()` / `is_linearizable_concave()`: DNLP composition rules
- `is_atom_smooth()`: atom-level smoothness (e.g., log, exp, sin)

### Smooth canonicalization (`cvxpy/reductions/dnlp2smooth/`)
`Dnlp2Smooth` converts DNLP expressions to smooth forms. Canonicalizers in `dnlp2smooth/canonicalizers/` handle atoms like log, exp, sin, power, geo_mean, etc. Registered in `SMOOTH_CANON_METHODS` dict.

### Diff engine (`cvxpy/reductions/solvers/nlp_solvers/diff_engine/`)
- `c_problem.py`: Wraps SparseDiffPy C library for AD (objective, gradient, Jacobian, Hessian)
- `converters.py`: Entry point — `convert_expr(expr, var_dict, n_vars, param_dict=None)` recursively converts CVXPY expressions to C diff engine nodes
- `registry.py`: `ATOM_CONVERTERS` dict mapping atom names to converter functions (40+ atoms)
- `helpers.py`: Shared utilities (`build_var_dict`, `build_param_dict`, matmul helpers, `normalize_shape`)

### Solver interfaces (`cvxpy/reductions/solvers/nlp_solvers/`)
- `nlp_solver.py`: Base `NLPsolver` class with `Bounds` (constraint extraction) and `Oracles` (diff engine wrapper)
- Solver implementations: `ipopt_nlpif.py`, `knitro_nlpif.py`, `uno_nlpif.py`, `copt_nlpif.py`
- Parameter updates flow through `Oracles.update_params()` → `C_problem.update_params()`

### Parameter support
Parameters are passed as param nodes to the diff engine at construction. On re-solve, `update_params()` updates values without rebuilding the expression graph. The solver cache (`_solver_cache['NLP']`) stores `Oracles` between solves.

## Implementing New Atoms

### 1. Create atom class in `cvxpy/atoms/` or `cvxpy/atoms/elementwise/`
Implement: `shape_from_args()`, `sign_from_args()`, `is_atom_convex()`, `is_atom_concave()`, `is_incr(idx)`, `is_decr(idx)`, `numeric(values)`.

### 2. Create DCP canonicalizer in `cvxpy/reductions/dcp2cone/canonicalizers/`
Register in `canonicalizers/__init__.py` → `CANON_METHODS[my_atom] = my_atom_canon`.

### 3. Create smooth canonicalizer in `cvxpy/reductions/dnlp2smooth/canonicalizers/` (if atom is smooth)
Register in `SMOOTH_CANON_METHODS`.

### 4. Add converter in `cvxpy/reductions/solvers/nlp_solvers/diff_engine/registry.py` → `ATOM_CONVERTERS` (for NLP support)

### 5. Export in `cvxpy/atoms/__init__.py`

## Testing

- Use `solver=cp.CLARABEL` for DCP tests that call `problem.solve()`
- NLP tests go in `cvxpy/tests/nlp_tests/`
- Test class inherits from `cvxpy.tests.base_test.BaseTest`
- Assertion helpers: `self.assertItemsAlmostEqual(a, b, places=5)`, `self.assertAlmostEqual(a, b, places=5)`
- Test with `Parameter` objects for DPP compliance

## Canon Backend Architecture

Backends handle matrix construction during `ConeMatrixStuffing`. Located in `cvxpy/lin_ops/backends/`.
- `CPP` (default) - C++ implementation, fastest for large expression trees
- `SCIPY` - Pure Python with SciPy sparse matrices
- `COO` - 3D COO tensor, better for large DPP problems with many parameters

Select via `CVXPY_DEFAULT_CANON_BACKEND=CPP` (default), `SCIPY`, or `COO`.

## Pull Requests

Always use the PR template in `.github/`. **Never check an item in the Contribution Checklist that has not actually been done.**

## Common Mistakes to Avoid

1. **Forgetting to register canonicalizers** in `canonicalizers/__init__.py` (both DCP and smooth)
2. **Forgetting to export atoms** in `cvxpy/atoms/__init__.py`
3. Missing `is_incr`/`is_decr` methods in atoms (breaks DCP analysis)
4. Not testing with `Parameter` objects (DPP compliance)
5. Missing license headers on new files
6. Not adding a diff engine converter for new atoms (breaks NLP support)
