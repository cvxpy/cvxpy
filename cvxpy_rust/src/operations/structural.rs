//! Structural operations: index, transpose, promote, broadcast_to, hstack, vstack, concatenate
//!
//! These operations transform the structure of tensors without arithmetic.

use crate::linop::{LinOp, LinOpData, SliceData, AxisSpec};
use crate::tensor::SparseTensor;
use super::{ProcessingContext, process_linop};

/// Process index operation
///
/// Selects rows from the tensor based on slice indices.
pub fn process_index(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Get slice data
    let slices = match &lin_op.data {
        LinOpData::Slices(s) => s,
        _ => panic!("Index operation must have slice data"),
    };

    // Process the argument
    let tensor = process_linop(&lin_op.args[0], ctx);

    // Compute the row indices to select
    let arg_shape = &lin_op.args[0].shape;
    let row_indices = compute_slice_indices(slices, arg_shape);

    // Select the rows
    tensor.select_rows(&row_indices)
}

/// Compute flat row indices from slice specifications
fn compute_slice_indices(slices: &[SliceData], shape: &[usize]) -> Vec<i64> {
    if slices.is_empty() {
        return vec![];
    }

    // Start with indices for first dimension
    let first_slice = &slices[0];
    let mut indices: Vec<i64> = (first_slice.start..first_slice.stop)
        .step_by(first_slice.step.max(1) as usize)
        .collect();

    // Cumulative product for multi-dimensional indexing
    let mut cum_prod = 1i64;

    for (dim, slice) in slices.iter().enumerate().skip(1) {
        let dim_size = shape.get(dim - 1).copied().unwrap_or(1) as i64;
        cum_prod *= dim_size;

        // Expand indices with new dimension
        let new_indices: Vec<i64> = (slice.start..slice.stop)
            .step_by(slice.step.max(1) as usize)
            .collect();

        let mut expanded = Vec::with_capacity(indices.len() * new_indices.len());
        for &base in &indices {
            for &new_idx in &new_indices {
                expanded.push(base + new_idx * cum_prod);
            }
        }
        indices = expanded;
    }

    indices
}

/// Process transpose operation
///
/// Permutes the rows of the tensor according to axis transposition.
pub fn process_transpose(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Get axes permutation
    let axes = match &lin_op.data {
        LinOpData::AxisData { axis: Some(AxisSpec::Multiple(axes)), .. } => axes.clone(),
        LinOpData::None => {
            // Default transpose: reverse all axes
            let n_dims = lin_op.args[0].shape.len();
            (0..n_dims).rev().map(|i| i as i64).collect()
        }
        _ => panic!("Transpose operation must have axis data"),
    };

    // Process the argument
    let tensor = process_linop(&lin_op.args[0], ctx);

    // Compute row permutation
    let original_shape = &lin_op.args[0].shape;
    let row_indices = compute_transpose_indices(original_shape, &axes);

    // Apply permutation by selecting rows
    tensor.select_rows(&row_indices)
}

/// Compute row indices for transposition
fn compute_transpose_indices(shape: &[usize], axes: &[i64]) -> Vec<i64> {
    let n = shape.iter().product::<usize>();
    if n == 0 {
        return vec![];
    }

    // Create index array and reshape/transpose/flatten
    let mut indices = Vec::with_capacity(n);

    // Build the transposed index mapping
    // For shape [a, b, c] with axes [1, 0, 2], we map
    // (i, j, k) in transposed -> (j, i, k) in original -> flat index

    let new_shape: Vec<usize> = axes.iter()
        .map(|&a| shape[a as usize])
        .collect();

    // Strides for original array (Fortran order)
    let mut orig_strides = vec![1i64; shape.len()];
    for i in 1..shape.len() {
        orig_strides[i] = orig_strides[i - 1] * shape[i - 1] as i64;
    }

    // Iterate over transposed array in Fortran order
    let mut transposed_idx = vec![0usize; new_shape.len()];
    for _ in 0..n {
        // Convert transposed index to original index
        let mut flat = 0i64;
        for (t_dim, &orig_dim) in axes.iter().enumerate() {
            flat += transposed_idx[t_dim] as i64 * orig_strides[orig_dim as usize];
        }
        indices.push(flat);

        // Increment transposed index (Fortran order)
        for dim in 0..new_shape.len() {
            transposed_idx[dim] += 1;
            if transposed_idx[dim] < new_shape[dim] {
                break;
            }
            transposed_idx[dim] = 0;
        }
    }

    indices
}

/// Process promote operation
///
/// Repeats a scalar value along rows.
pub fn process_promote(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Process the argument
    let tensor = process_linop(&lin_op.args[0], ctx);

    // Repeat rows to match output size
    let num_entries = lin_op.size();
    let row_indices: Vec<i64> = vec![0; num_entries];

    tensor.select_rows(&row_indices)
}

/// Process broadcast_to operation
///
/// Broadcasts tensor to a new shape by repeating elements.
pub fn process_broadcast_to(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Process the argument
    let tensor = process_linop(&lin_op.args[0], ctx);

    // Compute broadcast indices
    let original_shape = &lin_op.args[0].shape;
    let broadcast_shape = &lin_op.shape;
    let row_indices = compute_broadcast_indices(original_shape, broadcast_shape);

    tensor.select_rows(&row_indices)
}

/// Compute row indices for broadcasting
fn compute_broadcast_indices(original: &[usize], broadcast: &[usize]) -> Vec<i64> {
    let n_orig: usize = original.iter().product();
    let n_broadcast: usize = broadcast.iter().product();

    if n_orig == 0 {
        return vec![0; n_broadcast];
    }

    // Create flat indices for original array
    let mut orig_indices: Vec<i64> = (0..n_orig as i64).collect();

    // Reshape to original shape (conceptually)
    // Then broadcast to new shape
    // Then flatten

    // Simple case: scalar broadcast
    if n_orig == 1 {
        return vec![0; n_broadcast];
    }

    // Build mapping using NumPy-style broadcasting
    let mut result = Vec::with_capacity(n_broadcast);

    // Iterate over broadcast shape in Fortran order
    let mut broadcast_idx = vec![0usize; broadcast.len()];
    for _ in 0..n_broadcast {
        // Map broadcast index to original index
        let mut orig_flat = 0i64;
        let mut orig_stride = 1i64;

        // Align from the end (NumPy broadcasting rules)
        let offset = broadcast.len().saturating_sub(original.len());

        for (i, &dim_size) in original.iter().enumerate() {
            let broadcast_dim = i + offset;
            let idx = if dim_size == 1 {
                0  // Broadcast this dimension
            } else {
                broadcast_idx[broadcast_dim] % dim_size
            };
            orig_flat += idx as i64 * orig_stride;
            orig_stride *= dim_size as i64;
        }

        result.push(orig_flat);

        // Increment broadcast index (Fortran order)
        for dim in 0..broadcast.len() {
            broadcast_idx[dim] += 1;
            if broadcast_idx[dim] < broadcast[dim] {
                break;
            }
            broadcast_idx[dim] = 0;
        }
    }

    result
}

/// Process hstack operation
///
/// Horizontally stacks tensors (concatenate along axis 1 for 2D, axis 0 for 1D).
pub fn process_hstack(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    let total_rows = lin_op.size();
    let mut result = SparseTensor::with_capacity(
        (total_rows, ctx.var_length as usize + 1),
        lin_op.args.iter().map(|a| a.estimate_nnz()).sum(),
    );

    let mut row_offset = 0i64;
    for arg in &lin_op.args {
        let mut arg_tensor = process_linop(arg, ctx);
        let arg_rows = arg.size();

        // Offset the rows and extend result
        arg_tensor.offset_rows_in_place(row_offset);
        result.extend(arg_tensor);

        row_offset += arg_rows as i64;
    }

    result.shape = (total_rows, ctx.var_length as usize + 1);
    result
}

/// Process vstack operation
///
/// Vertically stacks tensors (concatenate along axis 0).
pub fn process_vstack(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    // First hstack, then permute rows
    let hstacked = process_hstack(lin_op, ctx);

    // Compute permutation for vstack
    let indices = compute_vstack_indices(&lin_op.args);

    hstacked.select_rows(&indices)
}

/// Compute row permutation for vstack
fn compute_vstack_indices(args: &[LinOp]) -> Vec<i64> {
    if args.is_empty() {
        return vec![];
    }

    let mut indices = Vec::new();
    let mut offset = 0i64;

    // Build index arrays for each arg
    let mut arg_indices: Vec<Vec<i64>> = Vec::new();
    for arg in args {
        let arg_rows = arg.size();
        let idx: Vec<i64> = (0..arg_rows as i64).map(|i| i + offset).collect();
        arg_indices.push(idx);
        offset += arg_rows as i64;
    }

    // Interleave according to vstack semantics
    // For vstack of [a, b, c], each with shape (m, n), the result has shape (3*m, n)
    // The order is: a[0], b[0], c[0], a[1], b[1], c[1], ...

    // Get shapes to understand the structure
    if args.iter().all(|a| a.shape.len() == 2) {
        // 2D case
        let rows_per_arg: Vec<usize> = args.iter().map(|a| a.shape[0]).collect();
        let cols = args[0].shape.get(1).copied().unwrap_or(1);

        for col in 0..cols {
            for (arg_idx, &n_rows) in rows_per_arg.iter().enumerate() {
                for row in 0..n_rows {
                    indices.push(arg_indices[arg_idx][row + col * n_rows]);
                }
            }
        }
    } else {
        // 1D or simple case - just concatenate
        for arg_idx in &arg_indices {
            indices.extend(arg_idx);
        }
    }

    indices
}

/// Process concatenate operation
///
/// Concatenates tensors along a specified axis.
pub fn process_concatenate(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    if lin_op.args.is_empty() {
        return SparseTensor::empty((lin_op.size(), ctx.var_length as usize + 1));
    }

    // Get axis
    let axis = match &lin_op.data {
        LinOpData::ConcatAxis(a) => *a,
        _ => None,
    };

    // First hstack all args
    let hstacked = process_hstack(lin_op, ctx);

    // If axis is None, flatten in C order
    if axis.is_none() {
        // For axis=None, arrays are flattened before concatenation
        // Since we already have flat representation, just return
        return hstacked;
    }

    let axis = axis.unwrap();

    // Compute permutation for concatenate along axis
    let indices = compute_concatenate_indices(&lin_op.args, axis);

    hstacked.select_rows(&indices)
}

/// Compute row permutation for concatenate along axis
fn compute_concatenate_indices(args: &[LinOp], axis: i64) -> Vec<i64> {
    if args.is_empty() {
        return vec![];
    }

    let mut indices = Vec::new();
    let mut offset = 0i64;

    // Build flat index arrays for each arg
    let arg_offsets: Vec<i64> = args.iter()
        .scan(0i64, |acc, arg| {
            let current = *acc;
            *acc += arg.size() as i64;
            Some(current)
        })
        .collect();

    // Get shapes and compute concatenation order
    let shapes: Vec<&Vec<usize>> = args.iter().map(|a| &a.shape).collect();

    if shapes.is_empty() || shapes[0].is_empty() {
        // Scalar case
        return arg_offsets;
    }

    let n_dims = shapes[0].len();
    let axis = if axis < 0 { n_dims as i64 + axis } else { axis } as usize;

    // Build the result shape
    let mut result_shape = shapes[0].clone();
    result_shape[axis] = shapes.iter().map(|s| s[axis]).sum();

    let total = result_shape.iter().product::<usize>();

    // Iterate over result indices in Fortran order
    let mut result_idx = vec![0usize; n_dims];
    for _ in 0..total {
        // Find which arg this belongs to
        let mut arg_idx = 0;
        let mut axis_offset = 0;
        for (i, shape) in shapes.iter().enumerate() {
            if result_idx[axis] < axis_offset + shape[axis] {
                arg_idx = i;
                break;
            }
            axis_offset += shape[axis];
        }

        // Compute flat index within the arg
        let mut local_idx = result_idx.clone();
        local_idx[axis] -= axis_offset;

        let mut flat = 0i64;
        let mut stride = 1i64;
        for dim in 0..n_dims {
            flat += local_idx[dim] as i64 * stride;
            stride *= shapes[arg_idx][dim] as i64;
        }

        indices.push(arg_offsets[arg_idx] + flat);

        // Increment result index (Fortran order)
        for dim in 0..n_dims {
            result_idx[dim] += 1;
            if result_idx[dim] < result_shape[dim] {
                break;
            }
            result_idx[dim] = 0;
        }
    }

    indices
}
