#![allow(unused_imports)]
#![allow(non_snake_case)]

use crate::linop::CvxpyShape;
use crate::linop::Linop;
use crate::linop::LinopKind;
use crate::view::Tensor;
use crate::view::View;
use crate::view::ViewContext;
use pyo3::prelude::*;

pub(crate) const CONST_ID: i64 = -1;

mod faer_ext;
mod linop;
mod tensor_representation;
mod tests;
mod view;

type SparseMatrix = faer::sparse::SparseColMat<u64, f64>;

type IdxMap = std::collections::HashMap<i64, i64>;

fn get_variable_tensor(shape: &CvxpyShape, id: i64) -> Tensor {
    assert!(id > CONST_ID);
    let n = shape.numel();
    return [(id, [(CONST_ID, faer_ext::eye(n))].into())].into();
}

pub(crate) fn process_constraints<'a>(linop: &Linop, view: View<'a>) -> View<'a> {
    match linop.kind {
        LinopKind::Variable(id) => View {
            variables: [id].into(),
            tensor: get_variable_tensor(&linop.shape, id),
            is_parameter_free: true,
            context: view.context,
        },
        _ => panic!(),
    }
}

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
