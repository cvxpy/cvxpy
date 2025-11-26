# cvxpy_rust

Rust canonicalization backend for CVXPY. Replaces the C++ cvxcore backend with a faster, safer Rust implementation.

## Overview

This crate converts CVXPY's LinOp expression trees into sparse coefficient matrices that optimization solvers can use. It's called during `problem.solve()` to build the `A`, `b`, and `c` matrices.

## Performance

Benchmarks show **~3x geometric mean speedup** over the C++ backend across various problem types and sizes.

## Building

Requires Rust 1.70+ and maturin.

```bash
cd cvxpy_rust

# Development build (fast, unoptimized)
maturin develop

# Release build (optimized)
maturin develop --release
```

## Usage

```python
import cvxpy as cp

x = cp.Variable(10)
prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= 0])

# Use Rust backend explicitly
prob.solve(canon_backend="RUST")

# Or set as default
cp.settings.CANON_BACKEND = "RUST"
prob.solve()
```

## Testing

### Rust Unit Tests

The crate has 27 native Rust unit tests covering tensor operations, matrix building, and all operation categories:

```bash
cd cvxpy_rust
cargo test
```

Tests are organized by module:
- `tensor.rs` - SparseTensor creation, manipulation, negation
- `matrix_builder.rs` - Single/multiple constraints, constants
- `operations/leaf.rs` - Variable, scalar_const, dense_const
- `operations/arithmetic.rs` - neg, sum, mul, mul_elem, div
- `operations/structural.rs` - transpose, reshape, hstack, vstack, index, promote
- `operations/specialized.rs` - sum_entries, trace, diag_vec, diag_mat, upper_tri

### Python Unit Tests

The Rust backend also has dedicated Python unit tests in `cvxpy/tests/test_python_backends.py`:

```bash
# Run Rust backend unit tests
pytest cvxpy/tests/test_python_backends.py::TestRustBackend -v
```

These tests compare the Rust backend's matrix output against the SciPy backend for correctness across 14 operations.

### Integration Tests

To verify the Rust backend works end-to-end:

```python
# Quick smoke test
import cvxpy as cp
import numpy as np

x = cp.Variable(5)
prob = cp.Problem(cp.Minimize(cp.norm(x - 1)), [x >= 0])
prob.solve(canon_backend="RUST")
print(f"Status: {prob.status}, x = {x.value}")
```

To run CVXPY's full test suite with Rust backend:

```bash
# Run a specific test file with Rust backend
CVXPY_CANON_BACKEND=RUST pytest cvxpy/tests/test_problem.py -v

# Or programmatically
import cvxpy as cp
cp.settings.CANON_BACKEND = "RUST"
# Then run tests...
```

## Architecture

```
src/
├── lib.rs           # PyO3 module entry point, build_matrix function
├── linop.rs         # LinOp struct with 22 operation types
├── tensor.rs        # SparseTensor (COO format) representation
├── matrix_builder.rs # Main canonicalization logic
└── operations/      # Implementation of each LinOp type
    ├── mod.rs
    ├── leaf.rs      # variable, scalar_const, dense_const, sparse_const, param
    ├── arithmetic.rs # mul, rmul, div, neg, mul_elem, sum
    ├── structural.rs # transpose, reshape, index, hstack, vstack, promote
    └── specialized.rs # trace, diag_vec, diag_mat, upper_tri, conv, kron, etc.
```

## How It Works

1. **Python calls `build_matrix()`** with a list of LinOp trees
2. **LinOps are extracted** from Python objects into Rust structs
3. **Each LinOp tree is processed recursively** to build sparse tensors
4. **Results are combined** into COO format (data, row, col arrays)
5. **NumPy arrays returned** to Python for scipy.sparse.csc_array

The GIL is released during the Rust computation (`py.allow_threads()`), allowing true parallelism with `rayon`.

## Supported Operations

All 22 LinOp types from CVXPY are supported:

| Category | Operations |
|----------|------------|
| Leaf | `variable`, `param`, `scalar_const`, `dense_const`, `sparse_const` |
| Arithmetic | `sum`, `neg`, `mul`, `rmul`, `div`, `mul_elem` |
| Structural | `transpose`, `reshape`, `index`, `hstack`, `vstack`, `promote` |
| Specialized | `sum_entries`, `trace`, `diag_vec`, `diag_mat`, `upper_tri`, `conv`, `kron_r`, `kron_l` |

## Dependencies

- `pyo3` - Python bindings
- `numpy` - NumPy array interop
- `ndarray` - N-dimensional arrays
- `sprs` - Sparse matrix operations
- `rayon` - Parallel processing
- `thiserror` - Error handling
