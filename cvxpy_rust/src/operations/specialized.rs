//! Specialized operations: sum_entries, trace, diag_vec, diag_mat, upper_tri, conv, kron_r, kron_l
//!
//! These operations perform specialized matrix/tensor transformations.

use crate::linop::{LinOp, LinOpData, AxisSpec};
use crate::tensor::SparseTensor;
use super::{ProcessingContext, process_linop};

/// Process sum_entries operation
///
/// Sums tensor entries along specified axis or all entries if axis is None.
pub fn process_sum_entries(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Get axis and keepdims from data
    let (axis, _keepdims) = match &lin_op.data {
        LinOpData::AxisData { axis, keepdims } => (axis.clone(), *keepdims),
        _ => (None, false),
    };

    // Process the argument
    let tensor = process_linop(&lin_op.args[0], ctx);
    let arg_shape = &lin_op.args[0].shape;

    match axis {
        None => {
            // Sum all entries - collapse all rows to row 0
            let mut result = SparseTensor::with_capacity(
                (1, ctx.var_length as usize + 1),
                tensor.nnz(),
            );
            for i in 0..tensor.nnz() {
                result.push(tensor.data[i], 0, tensor.cols[i], tensor.param_offsets[i]);
            }
            result
        }
        Some(axis_spec) => {
            // Sum along specific axis/axes
            let axes: Vec<i64> = match axis_spec {
                AxisSpec::Single(a) => vec![a],
                AxisSpec::Multiple(a) => a,
            };

            // Compute row mapping for sum along axes
            let row_mapping = compute_sum_row_mapping(arg_shape, &axes);
            let output_rows = lin_op.size();

            let mut result = SparseTensor::with_capacity(
                (output_rows, ctx.var_length as usize + 1),
                tensor.nnz(),
            );

            for i in 0..tensor.nnz() {
                let old_row = tensor.rows[i] as usize;
                if old_row < row_mapping.len() {
                    let new_row = row_mapping[old_row];
                    result.push(tensor.data[i], new_row, tensor.cols[i], tensor.param_offsets[i]);
                }
            }

            result
        }
    }
}

/// Compute row mapping for sum reduction along axes
fn compute_sum_row_mapping(shape: &[usize], axes: &[i64]) -> Vec<i64> {
    let n: usize = shape.iter().product();
    if n == 0 {
        return vec![];
    }

    let n_dims = shape.len();

    // Normalize negative axes
    let axes: Vec<usize> = axes.iter()
        .map(|&a| if a < 0 { (n_dims as i64 + a) as usize } else { a as usize })
        .collect();

    // Compute output shape (dimensions not in axes)
    let out_axes: Vec<bool> = (0..n_dims).map(|i| !axes.contains(&i)).collect();
    let out_dims: Vec<usize> = (0..n_dims)
        .filter(|&i| out_axes[i])
        .map(|i| shape[i])
        .collect();

    let out_n: usize = out_dims.iter().product();
    if out_n == 0 {
        return vec![0; n];
    }

    // Compute strides for output
    let mut out_strides = vec![1i64; out_dims.len()];
    for i in 1..out_dims.len() {
        out_strides[i] = out_strides[i - 1] * out_dims[i - 1] as i64;
    }

    // Map each input index to output index
    let mut mapping = Vec::with_capacity(n);

    // Input strides (Fortran order)
    let mut in_strides = vec![1i64; n_dims];
    for i in 1..n_dims {
        in_strides[i] = in_strides[i - 1] * shape[i - 1] as i64;
    }

    for flat_idx in 0..n {
        // Convert flat index to multi-index
        let mut remaining = flat_idx as i64;
        let mut multi_idx = vec![0usize; n_dims];
        for dim in (0..n_dims).rev() {
            multi_idx[dim] = (remaining / in_strides[dim]) as usize;
            remaining %= in_strides[dim];
        }

        // Project to output index
        let mut out_idx = 0i64;
        let mut out_dim = 0;
        for dim in 0..n_dims {
            if out_axes[dim] {
                out_idx += multi_idx[dim] as i64 * out_strides[out_dim];
                out_dim += 1;
            }
        }

        mapping.push(out_idx);
    }

    mapping
}

/// Process trace operation
///
/// Extracts the diagonal entries and sums them.
pub fn process_trace(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((1, ctx.var_length as usize + 1));
    }

    // Process the argument
    let tensor = process_linop(&lin_op.args[0], ctx);
    let arg_shape = &lin_op.args[0].shape;

    if arg_shape.len() < 2 {
        return tensor;
    }

    let n = arg_shape[0];  // Assumes square matrix

    // Diagonal indices for n x n matrix (flattened in Fortran order)
    // Diagonal entry (i, i) has flat index i + i*n = i*(n+1)
    let diag_indices: Vec<i64> = (0..n).map(|i| (i * (n + 1)) as i64).collect();

    // Select diagonal entries and sum to single row
    let mut result = SparseTensor::with_capacity(
        (1, ctx.var_length as usize + 1),
        tensor.nnz() / n.max(1),
    );

    for i in 0..tensor.nnz() {
        let row = tensor.rows[i];
        if diag_indices.contains(&row) {
            result.push(tensor.data[i], 0, tensor.cols[i], tensor.param_offsets[i]);
        }
    }

    result
}

/// Process diag_vec operation
///
/// Creates a diagonal matrix from a vector (adds zero rows for off-diagonal).
pub fn process_diag_vec(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Get diagonal offset k
    let k = match &lin_op.data {
        LinOpData::Int(k) => *k,
        _ => 0,
    };

    // Process the argument
    let tensor = process_linop(&lin_op.args[0], ctx);

    let n = lin_op.shape[0];  // Output is n x n
    let total_rows = n * n;

    let mut result = SparseTensor::with_capacity(
        (total_rows, ctx.var_length as usize + 1),
        tensor.nnz(),
    );

    // Map input row i to output row on diagonal
    for i in 0..tensor.nnz() {
        let input_row = tensor.rows[i] as usize;

        // Compute output row for diagonal position
        // For k=0: row i maps to flat index i*(n+1) = i + i*n
        // For k>0: row i maps to (i, i+k) -> i + (i+k)*n
        // For k<0: row i maps to (i-k, i) -> (i-k) + i*n
        let output_row = if k == 0 {
            input_row * (n + 1)
        } else if k > 0 {
            input_row + (input_row + k as usize) * n
        } else {
            (input_row + (-k) as usize) + input_row * n
        };

        if output_row < total_rows {
            result.push(
                tensor.data[i],
                output_row as i64,
                tensor.cols[i],
                tensor.param_offsets[i],
            );
        }
    }

    result
}

/// Process diag_mat operation
///
/// Extracts the diagonal from a matrix into a vector.
pub fn process_diag_mat(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Get diagonal offset k
    let k = match &lin_op.data {
        LinOpData::Int(k) => *k,
        _ => 0,
    };

    // Process the argument
    let tensor = process_linop(&lin_op.args[0], ctx);
    let arg_shape = &lin_op.args[0].shape;

    let rows = lin_op.shape[0];  // Output size
    let orig_rows = arg_shape.get(0).copied().unwrap_or(1);

    // Compute diagonal indices in the original matrix
    let diag_indices: Vec<i64> = (0..rows)
        .map(|i| {
            if k == 0 {
                (i * (orig_rows + 1)) as i64
            } else if k > 0 {
                (i + (i + k as usize) * orig_rows) as i64
            } else {
                ((i + (-k) as usize) + i * orig_rows) as i64
            }
        })
        .collect();

    // Build reverse mapping from original index to new index
    let mut result = SparseTensor::with_capacity(
        (rows, ctx.var_length as usize + 1),
        tensor.nnz() / orig_rows.max(1),
    );

    for i in 0..tensor.nnz() {
        let row = tensor.rows[i];
        if let Some(new_row) = diag_indices.iter().position(|&d| d == row) {
            result.push(
                tensor.data[i],
                new_row as i64,
                tensor.cols[i],
                tensor.param_offsets[i],
            );
        }
    }

    result
}

/// Process upper_tri operation
///
/// Extracts the upper triangular elements (excluding diagonal).
pub fn process_upper_tri(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Process the argument
    let tensor = process_linop(&lin_op.args[0], ctx);
    let arg_shape = &lin_op.args[0].shape;

    let n = arg_shape.get(0).copied().unwrap_or(1);

    // Compute upper triangular indices (k=1, excluding diagonal)
    let mut upper_indices = Vec::new();
    for j in 1..n {
        for i in 0..j {
            upper_indices.push((i + j * n) as i64);  // Fortran order
        }
    }

    // Select upper triangular entries and renumber
    let mut result = SparseTensor::with_capacity(
        (upper_indices.len(), ctx.var_length as usize + 1),
        tensor.nnz() / 2,
    );

    for i in 0..tensor.nnz() {
        let row = tensor.rows[i];
        if let Some(new_row) = upper_indices.iter().position(|&u| u == row) {
            result.push(
                tensor.data[i],
                new_row as i64,
                tensor.cols[i],
                tensor.param_offsets[i],
            );
        }
    }

    result
}

/// Process conv operation
///
/// Constructs a Toeplitz matrix for convolution.
pub fn process_conv(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Get convolution kernel from data
    let kernel_linop = match &lin_op.data {
        LinOpData::LinOpRef(inner) => inner.as_ref(),
        _ => panic!("Conv operation must have LinOp data (kernel)"),
    };

    // Process the argument (signal)
    let tensor = process_linop(&lin_op.args[0], ctx);

    // Extract kernel data
    let kernel = get_kernel_data(kernel_linop);
    let kernel_len = kernel.len();

    let arg_len = lin_op.args[0].size();
    let output_len = lin_op.size();

    // Build Toeplitz matrix multiplication
    let mut result = SparseTensor::with_capacity(
        (output_len, ctx.var_length as usize + 1),
        tensor.nnz() * kernel_len,
    );

    // For each entry in tensor, apply convolution
    for i in 0..tensor.nnz() {
        let input_row = tensor.rows[i] as usize;

        // Convolve with kernel
        for (k_idx, &k_val) in kernel.iter().enumerate() {
            if k_val != 0.0 {
                let output_row = input_row + k_idx;
                if output_row < output_len {
                    result.push(
                        tensor.data[i] * k_val,
                        output_row as i64,
                        tensor.cols[i],
                        tensor.param_offsets[i],
                    );
                }
            }
        }
    }

    result
}

/// Extract kernel data from LinOp
fn get_kernel_data(lin_op: &LinOp) -> Vec<f64> {
    match &lin_op.data {
        LinOpData::Float(v) => vec![*v],
        LinOpData::Int(v) => vec![*v as f64],
        LinOpData::DenseArray { data, .. } => data.clone(),
        LinOpData::SparseArray { data, indices, indptr, shape } => {
            // Convert to dense
            let n = shape.0 * shape.1;
            let mut dense = vec![0.0; n];
            let n_cols = indptr.len() - 1;
            for j in 0..n_cols {
                let start = indptr[j] as usize;
                let end = indptr[j + 1] as usize;
                for idx in start..end {
                    let i = indices[idx] as usize;
                    dense[j * shape.0 + i] = data[idx];
                }
            }
            dense
        }
        _ => vec![1.0],
    }
}

/// Process kron_r operation
///
/// Kronecker product: data ⊗ arg
pub fn process_kron_r(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Get left operand data
    let lhs_linop = match &lin_op.data {
        LinOpData::LinOpRef(inner) => inner.as_ref(),
        _ => panic!("KronR operation must have LinOp data"),
    };

    // Process the argument (right operand)
    let rhs = process_linop(&lin_op.args[0], ctx);

    // Extract left operand data
    let lhs_data = get_kernel_data(lhs_linop);
    let lhs_shape = &lhs_linop.shape;
    let rhs_shape = &lin_op.args[0].shape;

    // Compute Kronecker product indices
    let (row_indices, scale_factors) = compute_kron_indices(&lhs_data, lhs_shape, rhs_shape);

    let output_rows = lin_op.size();
    let mut result = SparseTensor::with_capacity(
        (output_rows, ctx.var_length as usize + 1),
        rhs.nnz() * lhs_data.iter().filter(|&&x| x != 0.0).count(),
    );

    // Apply Kronecker product
    for i in 0..rhs.nnz() {
        let rhs_row = rhs.rows[i] as usize;

        for (lhs_idx, &lhs_val) in lhs_data.iter().enumerate() {
            if lhs_val != 0.0 {
                let new_row = row_indices[lhs_idx * rhs.shape.0 + rhs_row];
                result.push(
                    rhs.data[i] * lhs_val,
                    new_row,
                    rhs.cols[i],
                    rhs.param_offsets[i],
                );
            }
        }
    }

    result
}

/// Process kron_l operation
///
/// Kronecker product: arg ⊗ data
pub fn process_kron_l(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Get right operand data
    let rhs_linop = match &lin_op.data {
        LinOpData::LinOpRef(inner) => inner.as_ref(),
        _ => panic!("KronL operation must have LinOp data"),
    };

    // Process the argument (left operand)
    let lhs = process_linop(&lin_op.args[0], ctx);

    // Extract right operand data
    let rhs_data = get_kernel_data(rhs_linop);
    let lhs_shape = &lin_op.args[0].shape;
    let rhs_shape = &rhs_linop.shape;

    // Compute Kronecker product indices
    let (row_indices, _) = compute_kron_indices_l(lhs_shape, &rhs_data, rhs_shape);

    let output_rows = lin_op.size();
    let rhs_size: usize = rhs_shape.iter().product();

    let mut result = SparseTensor::with_capacity(
        (output_rows, ctx.var_length as usize + 1),
        lhs.nnz() * rhs_data.iter().filter(|&&x| x != 0.0).count(),
    );

    // Apply Kronecker product
    for i in 0..lhs.nnz() {
        let lhs_row = lhs.rows[i] as usize;

        for (rhs_idx, &rhs_val) in rhs_data.iter().enumerate() {
            if rhs_val != 0.0 {
                let new_row = row_indices[lhs_row * rhs_size + rhs_idx];
                result.push(
                    lhs.data[i] * rhs_val,
                    new_row,
                    lhs.cols[i],
                    lhs.param_offsets[i],
                );
            }
        }
    }

    result
}

/// Compute Kronecker product row indices for kron(A, B)
fn compute_kron_indices(lhs_data: &[f64], lhs_shape: &[usize], rhs_shape: &[usize]) -> (Vec<i64>, Vec<f64>) {
    let lhs_size: usize = lhs_shape.iter().product();
    let rhs_size: usize = rhs_shape.iter().product();
    let total = lhs_size * rhs_size;

    let mut row_indices = Vec::with_capacity(total);
    let mut scale_factors = Vec::with_capacity(total);

    // kron(A, B)[i,j] = A[i//b_rows, j//b_cols] * B[i%b_rows, j%b_cols]
    // In Fortran order, we iterate rhs fast, lhs slow

    for lhs_idx in 0..lhs_size {
        for rhs_idx in 0..rhs_size {
            let new_row = (lhs_idx * rhs_size + rhs_idx) as i64;
            row_indices.push(new_row);
            scale_factors.push(lhs_data[lhs_idx]);
        }
    }

    (row_indices, scale_factors)
}

/// Compute Kronecker product row indices for kron(A, B) where A is variable
fn compute_kron_indices_l(lhs_shape: &[usize], rhs_data: &[f64], rhs_shape: &[usize]) -> (Vec<i64>, Vec<f64>) {
    let lhs_size: usize = lhs_shape.iter().product();
    let rhs_size: usize = rhs_shape.iter().product();
    let total = lhs_size * rhs_size;

    let mut row_indices = Vec::with_capacity(total);
    let mut scale_factors = Vec::with_capacity(total);

    for lhs_idx in 0..lhs_size {
        for rhs_idx in 0..rhs_size {
            let new_row = (lhs_idx * rhs_size + rhs_idx) as i64;
            row_indices.push(new_row);
            scale_factors.push(rhs_data[rhs_idx]);
        }
    }

    (row_indices, scale_factors)
}
