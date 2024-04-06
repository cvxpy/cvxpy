#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use crate::view::ViewContext;
use pyo3::prelude::*;

mod backend;
mod faer_ext;
mod linop;
mod tensor_representation;
mod tests;
mod view;

type SparseMatrix = faer::sparse::SparseColMat<u64, f64>;
type NdArray = ndarray::Array2<f64>;

type IdxMap = std::collections::HashMap<i64, i64>;

#[pyfunction]
fn build_matrix(
    id_to_col: IdxMap,
    param_to_size: IdxMap,
    param_to_col: IdxMap,
    param_size_plus_one: i64,
    var_length: i64,
    linops: Vec<linop::Linop>,
) -> PyResult<PyObject> {
    let ctx = ViewContext {
        id_to_col,
        param_to_size,
        param_to_col,
        param_size_plus_one,
        var_length,
    };

    todo!();
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_matrix, m)?)?;
    Ok(())
}
