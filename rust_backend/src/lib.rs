#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use crate::view::{ViewContext, View};
use crate::backend::process_constraints;
use crate::tensor_representation::TensorRepresentation;
use pyo3::prelude::*;

mod backend;
mod faer_ext;
mod linop;
mod tensor_representation;
mod tests;
mod view;

type SparseMatrix = faer::sparse::SparseColMat<u64, f64>;
type NdArray = numpy::ndarray::ArrayView2<'static, f64>;

type IdxMap = std::collections::HashMap<i64, i64>;

#[pyfunction]
fn build_matrix(
    mut id_to_col: IdxMap,
    param_to_size: IdxMap,
    param_to_col: IdxMap,
    param_size_plus_one: i64,
    var_length: i64,
    linops: Vec<linop::Linop>,
) -> PyResult<(Vec<f64>, (Vec<u64>, Vec<u64>), (u64, u64))> {
    id_to_col.insert(-1, var_length); // May do this in Python to remove mut
    let ctx = ViewContext {
        id_to_col,
        param_to_size,
        param_to_col,
        param_size_plus_one,
        var_length,
    };

    let mut offset = 0i64;
    let tensor = TensorRepresentation::combine(linops.into_iter().map(|linop| {
        let view = View::new(&ctx);
        let lin_op_tensor = process_constraints(&linop, view);
        let tensor_rep = lin_op_tensor.get_tensor_representation(offset);
        offset += i64::try_from(linop.shape.numel()).unwrap();
        tensor_rep
    }).collect());

    Ok(transform_tensor_to_mat(&ctx, tensor, offset.try_into().unwrap()))

}

fn transform_tensor_to_mat(ctx: &ViewContext, tensor: TensorRepresentation, offset: u64) -> (Vec<f64>, (Vec<u64>, Vec<u64>), (u64, u64)) {
    let rows = tensor.col.into_iter().zip(tensor.row.into_iter()).map(|(c, r)| c * offset + r).collect();
    let shape = (offset * u64::try_from(ctx.var_length + 1).unwrap(), ctx.param_size_plus_one.try_into().unwrap());

    (tensor.data, (rows, tensor.parameter_offset), shape)
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_matrix, m)?)?;
    Ok(())
}
