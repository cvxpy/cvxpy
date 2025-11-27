//! cvxpy_rust - Rust canonicalization backend for CVXPY
//!
//! This crate provides a high-performance replacement for the C++ cvxcore backend.
//! It converts LinOp trees into sparse matrices for optimization solvers.

// Allow some clippy lints that are too noisy for this codebase
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::useless_conversion)] // False positives from PyO3 macro expansion

mod linop;
mod matrix_builder;
mod operations;
mod tensor;

use numpy::PyArray1;
use numpy::ToPyArray;
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::linop::LinOp;
use crate::matrix_builder::build_matrix_internal;

/// Build the coefficient matrix from LinOp trees.
///
/// This is the main entry point called from Python's RustCanonBackend.
///
/// # Arguments
/// * `lin_ops` - List of LinOp trees representing constraints/objective
/// * `param_size_plus_one` - Number of parameter slices plus one for constants
/// * `id_to_col` - Maps variable IDs to column offsets
/// * `param_to_size` - Maps parameter IDs to their sizes
/// * `param_to_col` - Maps parameter IDs to column offsets in param vector
/// * `var_length` - Total number of variables
///
/// # Returns
/// Tuple of (data, (row, col), shape) in COO format for scipy.sparse.csc_array
#[pyfunction]
fn build_matrix<'py>(
    py: Python<'py>,
    lin_ops: Vec<Bound<'py, PyAny>>,
    param_size_plus_one: i64,
    id_to_col: HashMap<i64, i64>,
    param_to_size: HashMap<i64, i64>,
    param_to_col: HashMap<i64, i64>,
    var_length: i64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>),
    (i64, i64),
)> {
    // Extract LinOp trees from Python objects
    let rust_lin_ops: Vec<LinOp> = lin_ops
        .iter()
        .map(|obj| LinOp::from_python(obj))
        .collect::<PyResult<Vec<_>>>()?;

    // Build the matrix (release GIL during computation)
    let result = py.allow_threads(|| {
        build_matrix_internal(
            &rust_lin_ops,
            param_size_plus_one,
            &id_to_col,
            &param_to_size,
            &param_to_col,
            var_length,
        )
    });

    // Convert to numpy arrays
    let data = result.data.to_pyarray_bound(py);
    let rows = result.rows.to_pyarray_bound(py);
    let cols = result.cols.to_pyarray_bound(py);
    let shape = (result.shape.0 as i64, result.shape.1 as i64);

    Ok((data, (rows, cols), shape))
}

/// Python module definition
#[pymodule]
fn cvxpy_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_matrix, m)?)?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
