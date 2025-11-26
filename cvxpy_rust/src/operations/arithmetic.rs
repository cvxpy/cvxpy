//! Arithmetic operations: neg, mul, rmul, mul_elem, div
//!
//! These operations perform arithmetic transformations on tensors.

use crate::linop::{LinOp, LinOpData};
use crate::tensor::SparseTensor;
use super::{ProcessingContext, process_linop};

/// Process negation operation
///
/// Negates all values in the tensor: (A, b) -> (-A, -b)
pub fn process_neg(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    let mut tensor = process_linop(&lin_op.args[0], ctx);
    tensor.negate_in_place();
    tensor
}

/// Process left multiplication: data @ arg
///
/// Multiplies the argument tensor from the left by a constant matrix.
/// The constant matrix is block-diagonalized according to the output shape.
pub fn process_mul(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Get the constant data (lhs)
    let lhs_linop = match &lin_op.data {
        LinOpData::LinOpRef(inner) => inner.as_ref(),
        _ => panic!("Mul operation must have LinOp data"),
    };

    // Process the argument (rhs)
    let rhs = process_linop(&lin_op.args[0], ctx);

    // Get constant data tensor
    let lhs_data = get_constant_matrix_data(lhs_linop);

    // Perform block diagonal multiplication
    multiply_block_diagonal(&lhs_data, &rhs, lin_op, ctx, false)
}

/// Process right multiplication: arg @ data
///
/// Multiplies the argument tensor from the right by a constant matrix.
pub fn process_rmul(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Get the constant data (rhs in mathematical sense)
    let rhs_linop = match &lin_op.data {
        LinOpData::LinOpRef(inner) => inner.as_ref(),
        _ => panic!("Rmul operation must have LinOp data"),
    };

    // Process the argument (lhs in mathematical sense)
    let lhs = process_linop(&lin_op.args[0], ctx);

    // Get constant data tensor
    let rhs_data = get_constant_matrix_data(rhs_linop);

    // Perform block diagonal multiplication from right
    multiply_block_diagonal_right(&lhs, &rhs_data, lin_op, ctx)
}

/// Process elementwise multiplication: arg * data
///
/// Multiplies the argument tensor elementwise by constant data.
/// This is equivalent to left multiplication by a diagonal matrix.
pub fn process_mul_elem(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Get the constant data
    let data_linop = match &lin_op.data {
        LinOpData::LinOpRef(inner) => inner.as_ref(),
        _ => panic!("MulElem operation must have LinOp data"),
    };

    // Process the argument
    let mut tensor = process_linop(&lin_op.args[0], ctx);

    // Get constant data as flat array
    let data = get_constant_vector_data(data_linop);

    // Multiply each tensor entry by the corresponding data element
    for i in 0..tensor.nnz() {
        let row = tensor.rows[i] as usize;
        if row < data.len() {
            tensor.data[i] *= data[row];
        }
    }

    tensor
}

/// Process division: arg / data
///
/// Divides the argument tensor elementwise by constant data.
/// This is equivalent to elementwise multiplication by reciprocal.
pub fn process_div(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Get the constant data
    let data_linop = match &lin_op.data {
        LinOpData::LinOpRef(inner) => inner.as_ref(),
        _ => panic!("Div operation must have LinOp data"),
    };

    // Process the argument
    let mut tensor = process_linop(&lin_op.args[0], ctx);

    // Get constant data as flat array
    let data = get_constant_vector_data(data_linop);

    // Divide each tensor entry by the corresponding data element
    for i in 0..tensor.nnz() {
        let row = tensor.rows[i] as usize;
        if row < data.len() && data[row] != 0.0 {
            tensor.data[i] /= data[row];
        }
    }

    tensor
}

/// Helper: extract constant matrix data from a LinOp (2D case)
/// For left multiplication, 1D arrays are treated as row vectors (1, n)
fn get_constant_matrix_data(lin_op: &LinOp) -> ConstantMatrix {
    match &lin_op.data {
        LinOpData::Float(v) => ConstantMatrix::Scalar(*v),
        LinOpData::Int(v) => ConstantMatrix::Scalar(*v as f64),
        LinOpData::DenseArray { data, shape } => {
            if shape.is_empty() || shape.iter().product::<usize>() == 1 {
                ConstantMatrix::Scalar(data[0])
            } else if shape.len() == 1 {
                // 1D array treated as row vector (1, n) for left multiplication
                // This matches Python: lin_op_shape = [1, lin_op.shape[0]]
                ConstantMatrix::Dense {
                    data: data.clone(),
                    rows: 1,
                    cols: shape[0],
                }
            } else {
                ConstantMatrix::Dense {
                    data: data.clone(),
                    rows: shape[0],
                    cols: shape[1],
                }
            }
        }
        LinOpData::SparseArray { data, indices, indptr, shape } => {
            ConstantMatrix::Sparse {
                values: data.clone(),
                row_indices: indices.clone(),
                col_indptr: indptr.clone(),
                rows: shape.0,
                cols: shape.1,
            }
        }
        _ => panic!("Cannot extract constant matrix from {:?}", lin_op.op_type),
    }
}

/// Helper: extract constant vector data from a LinOp (flattened)
fn get_constant_vector_data(lin_op: &LinOp) -> Vec<f64> {
    match &lin_op.data {
        LinOpData::Float(v) => vec![*v],
        LinOpData::Int(v) => vec![*v as f64],
        LinOpData::DenseArray { data, .. } => data.clone(),
        LinOpData::SparseArray { data, indices, indptr, shape } => {
            // Convert sparse to dense for elementwise ops
            let n = shape.0 * shape.1;
            let mut dense = vec![0.0; n];
            let n_cols = indptr.len() - 1;
            for j in 0..n_cols {
                let start = indptr[j] as usize;
                let end = indptr[j + 1] as usize;
                for idx in start..end {
                    let i = indices[idx] as usize;
                    let flat_idx = j * shape.0 + i;
                    dense[flat_idx] = data[idx];
                }
            }
            dense
        }
        _ => panic!("Cannot extract constant vector from {:?}", lin_op.op_type),
    }
}

/// Representation of constant matrix data
#[derive(Debug)]
enum ConstantMatrix {
    Scalar(f64),
    Dense { data: Vec<f64>, rows: usize, cols: usize },
    Sparse {
        values: Vec<f64>,
        row_indices: Vec<i64>,
        col_indptr: Vec<i64>,
        rows: usize,
        cols: usize,
    },
}

/// Block diagonal multiplication from left: kron(I, A) @ tensor
fn multiply_block_diagonal(
    lhs: &ConstantMatrix,
    rhs: &SparseTensor,
    lin_op: &LinOp,
    ctx: &ProcessingContext,
    _transpose_lhs: bool,
) -> SparseTensor {
    let output_rows = lin_op.size();

    match lhs {
        ConstantMatrix::Scalar(s) => {
            // Scalar multiplication
            let mut result = rhs.clone();
            result.scale_in_place(*s);
            result.shape = (output_rows, ctx.var_length as usize + 1);
            result
        }
        ConstantMatrix::Dense { data, rows, cols } => {
            // Dense block diagonal multiplication
            multiply_dense_block_diagonal(data, *rows, *cols, rhs, output_rows, ctx)
        }
        ConstantMatrix::Sparse { values, row_indices, col_indptr, rows, cols } => {
            // Sparse block diagonal multiplication
            multiply_sparse_block_diagonal(
                values, row_indices, col_indptr,
                *rows, *cols, rhs, output_rows, ctx
            )
        }
    }
}

/// Dense block diagonal multiplication: kron(I_k, A) @ tensor
fn multiply_dense_block_diagonal(
    data: &[f64],
    a_rows: usize,
    a_cols: usize,
    rhs: &SparseTensor,
    output_rows: usize,
    ctx: &ProcessingContext,
) -> SparseTensor {
    // Number of blocks
    let k = if a_cols > 0 { rhs.shape.0 / a_cols } else { 1 };

    // Estimate output nnz
    let est_nnz = rhs.nnz() * a_rows;
    let mut result = SparseTensor::with_capacity(
        (output_rows, ctx.var_length as usize + 1),
        est_nnz,
    );

    // For each entry in rhs, compute its contribution after multiplication
    for idx in 0..rhs.nnz() {
        let rhs_row = rhs.rows[idx] as usize;
        let rhs_col = rhs.cols[idx];
        let rhs_val = rhs.data[idx];
        let rhs_param = rhs.param_offsets[idx];

        // Determine which block this entry belongs to
        let block = rhs_row / a_cols;
        let col_in_block = rhs_row % a_cols;

        // Apply A to this entry
        for i in 0..a_rows {
            let a_val = data[col_in_block * a_rows + i];  // Column-major
            if a_val != 0.0 {
                let new_row = (block * a_rows + i) as i64;
                result.push(a_val * rhs_val, new_row, rhs_col, rhs_param);
            }
        }
    }

    result
}

/// Sparse block diagonal multiplication: kron(I_k, A) @ tensor
fn multiply_sparse_block_diagonal(
    values: &[f64],
    row_indices: &[i64],
    col_indptr: &[i64],
    a_rows: usize,
    a_cols: usize,
    rhs: &SparseTensor,
    output_rows: usize,
    ctx: &ProcessingContext,
) -> SparseTensor {
    // Number of blocks
    let k = if a_cols > 0 { rhs.shape.0 / a_cols } else { 1 };

    // Estimate output nnz
    let est_nnz = rhs.nnz() * values.len() / a_cols.max(1);
    let mut result = SparseTensor::with_capacity(
        (output_rows, ctx.var_length as usize + 1),
        est_nnz,
    );

    // For each entry in rhs, compute its contribution
    for idx in 0..rhs.nnz() {
        let rhs_row = rhs.rows[idx] as usize;
        let rhs_col = rhs.cols[idx];
        let rhs_val = rhs.data[idx];
        let rhs_param = rhs.param_offsets[idx];

        // Determine which block and column in block
        let block = rhs_row / a_cols;
        let col_in_block = rhs_row % a_cols;

        // Get non-zeros in column col_in_block of A
        if col_in_block < col_indptr.len() - 1 {
            let start = col_indptr[col_in_block] as usize;
            let end = col_indptr[col_in_block + 1] as usize;

            for nnz_idx in start..end {
                let a_row = row_indices[nnz_idx] as usize;
                let a_val = values[nnz_idx];
                if a_val != 0.0 {
                    let new_row = (block * a_rows + a_row) as i64;
                    result.push(a_val * rhs_val, new_row, rhs_col, rhs_param);
                }
            }
        }
    }

    result
}

/// Block diagonal multiplication from right: tensor @ kron(I_k, A^T)
fn multiply_block_diagonal_right(
    lhs: &SparseTensor,
    rhs: &ConstantMatrix,
    lin_op: &LinOp,
    ctx: &ProcessingContext,
) -> SparseTensor {
    let output_rows = lin_op.size();

    match rhs {
        ConstantMatrix::Scalar(s) => {
            // Scalar multiplication
            let mut result = lhs.clone();
            result.scale_in_place(*s);
            result.shape = (output_rows, ctx.var_length as usize + 1);
            result
        }
        ConstantMatrix::Dense { data, rows, cols } => {
            // Dense block diagonal right multiplication
            // kron(A^T, I_k) in terms of block structure
            multiply_dense_block_diagonal_right(data, *rows, *cols, lhs, output_rows, ctx)
        }
        ConstantMatrix::Sparse { values, row_indices, col_indptr, rows, cols } => {
            // Sparse block diagonal right multiplication
            multiply_sparse_block_diagonal_right(
                values, row_indices, col_indptr,
                *rows, *cols, lhs, output_rows, ctx
            )
        }
    }
}

/// Dense block diagonal right multiplication
fn multiply_dense_block_diagonal_right(
    data: &[f64],
    a_rows: usize,
    a_cols: usize,
    lhs: &SparseTensor,
    output_rows: usize,
    ctx: &ProcessingContext,
) -> SparseTensor {
    // Number of blocks
    let k = if a_rows > 0 { lhs.shape.0 / a_rows } else { 1 };

    let est_nnz = lhs.nnz() * a_cols;
    let mut result = SparseTensor::with_capacity(
        (output_rows, ctx.var_length as usize + 1),
        est_nnz,
    );

    // For each entry in lhs, compute contribution
    for idx in 0..lhs.nnz() {
        let lhs_row = lhs.rows[idx] as usize;
        let lhs_col = lhs.cols[idx];
        let lhs_val = lhs.data[idx];
        let lhs_param = lhs.param_offsets[idx];

        // Determine which block and row in block
        let block = lhs_row / a_rows;
        let row_in_block = lhs_row % a_rows;

        // Multiply by each column of A^T (row of A)
        for j in 0..a_cols {
            let a_val = data[j * a_rows + row_in_block];  // A[row_in_block, j] in column-major
            if a_val != 0.0 {
                let new_row = (block * a_cols + j) as i64;
                result.push(a_val * lhs_val, new_row, lhs_col, lhs_param);
            }
        }
    }

    result
}

/// Sparse block diagonal right multiplication
fn multiply_sparse_block_diagonal_right(
    values: &[f64],
    row_indices: &[i64],
    col_indptr: &[i64],
    a_rows: usize,
    a_cols: usize,
    lhs: &SparseTensor,
    output_rows: usize,
    ctx: &ProcessingContext,
) -> SparseTensor {
    // Build row-indexed structure for efficient access
    // We need to find all entries in row row_in_block of A
    // CSC stores columns, so we need to iterate all columns

    let est_nnz = lhs.nnz() * values.len() / a_rows.max(1);
    let mut result = SparseTensor::with_capacity(
        (output_rows, ctx.var_length as usize + 1),
        est_nnz,
    );

    // For each entry in lhs
    for idx in 0..lhs.nnz() {
        let lhs_row = lhs.rows[idx] as usize;
        let lhs_col = lhs.cols[idx];
        let lhs_val = lhs.data[idx];
        let lhs_param = lhs.param_offsets[idx];

        let block = lhs_row / a_rows;
        let row_in_block = lhs_row % a_rows;

        // Find all (row_in_block, j) entries in A
        for j in 0..a_cols {
            if j < col_indptr.len() - 1 {
                let start = col_indptr[j] as usize;
                let end = col_indptr[j + 1] as usize;

                for nnz_idx in start..end {
                    if row_indices[nnz_idx] as usize == row_in_block {
                        let a_val = values[nnz_idx];
                        if a_val != 0.0 {
                            let new_row = (block * a_cols + j) as i64;
                            result.push(a_val * lhs_val, new_row, lhs_col, lhs_param);
                        }
                    }
                }
            }
        }
    }

    result
}
