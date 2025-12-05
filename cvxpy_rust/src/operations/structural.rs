//! Structural operations: index, transpose, promote, broadcast_to, hstack, vstack, concatenate
//!
//! These operations transform the structure of tensors without arithmetic.

use super::{process_linop, ProcessingContext};
use crate::linop::{AxisSpec, LinOp, LinOpData, SliceData};
use crate::tensor::SparseTensor;

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
///
/// Uses column-major (Fortran) order to match SciPy's behavior.
/// For a 2D array with shape (m, n) and slices [slice0, slice1]:
///   flat_index = i + j * m  (column-major)
/// where i is from slice0 and j is from slice1.
/// Generate indices for a slice, handling negative steps
fn slice_indices(start: i64, stop: i64, step: i64) -> Vec<i64> {
    if step > 0 {
        (start..stop).step_by(step as usize).collect()
    } else if step < 0 {
        // For negative step, we iterate backwards
        let mut result = Vec::new();
        let mut i = start;
        while i > stop {
            result.push(i);
            i += step; // step is negative, so this decreases i
        }
        result
    } else {
        // step == 0 is invalid, return empty
        vec![]
    }
}

fn compute_slice_indices(slices: &[SliceData], shape: &[usize]) -> Vec<i64> {
    if slices.is_empty() {
        return vec![];
    }

    // Start with indices for first dimension
    let first_slice = &slices[0];
    let mut indices = slice_indices(first_slice.start, first_slice.stop, first_slice.step);

    // Cumulative product for multi-dimensional indexing (column-major)
    let mut cum_prod = 1i64;

    for (dim, slice) in slices.iter().enumerate().skip(1) {
        let dim_size = shape.get(dim - 1).copied().unwrap_or(1) as i64;
        cum_prod *= dim_size;

        // Expand indices with new dimension
        let new_indices = slice_indices(slice.start, slice.stop, slice.step);

        // Important: iterate in Fortran order (new_indices outer, indices inner)
        // This matches np.add.outer(rows, new_indices * cum_prod).flatten(order="F")
        let mut expanded = Vec::with_capacity(indices.len() * new_indices.len());
        for &new_idx in &new_indices {
            for &base in &indices {
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
        LinOpData::AxisData {
            axis: Some(AxisSpec::Multiple(axes)),
            ..
        } => axes.clone(),
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

    let new_shape: Vec<usize> = axes.iter().map(|&a| shape[a as usize]).collect();

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

    // Create flat indices for original array (unused but documents algorithm)
    let _orig_indices: Vec<i64> = (0..n_orig as i64).collect();

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
            let idx = if dim_size <= 1 {
                0 // Broadcast this dimension (size 0 or 1)
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
    let indices = compute_vstack_indices(&lin_op.args, &lin_op.shape);

    hstacked.select_rows(&indices)
}

/// Compute row permutation for vstack
fn compute_vstack_indices(args: &[LinOp], output_shape: &[usize]) -> Vec<i64> {
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

    // Check if output is 2D - need column-major ordering
    if output_shape.len() == 2 {
        // Vstacking creates a 2D result
        // For vstack of 1D arrays [a, b] and [c, d] into (2, 2):
        //   Output in F-order: (0,0), (1,0), (0,1), (1,1) = a, c, b, d
        // For vstack of 2D arrays (m, n) each into (k*m, n):
        //   Interleave rows within each column

        let _n_rows_output = output_shape[0]; // unused but documents layout
        let n_cols_output = output_shape[1];
        let _n_args = args.len(); // unused but documents layout

        // How many rows does each arg contribute?
        let rows_per_arg: Vec<usize> = args
            .iter()
            .map(|a| if a.shape.len() == 2 { a.shape[0] } else { 1 })
            .collect();

        // Columns per arg (should all be equal or 1 for broadcasting)
        let cols_per_arg: Vec<usize> = args
            .iter()
            .map(|a| {
                if a.shape.len() == 2 {
                    a.shape[1]
                } else {
                    a.size()
                }
            })
            .collect();

        // Iterate over output in Fortran order (column by column)
        for col in 0..n_cols_output {
            for (arg_idx, &n_rows) in rows_per_arg.iter().enumerate() {
                let _arg_cols = cols_per_arg[arg_idx]; // unused but documents layout
                for row in 0..n_rows {
                    // Index into the arg's flat representation
                    let arg_flat_idx = if args[arg_idx].shape.len() == 2 {
                        row + col * n_rows
                    } else {
                        // 1D arg: col is the index
                        col
                    };
                    // Bounds check to prevent panic
                    if arg_flat_idx < arg_indices[arg_idx].len() {
                        indices.push(arg_indices[arg_idx][arg_flat_idx]);
                    }
                }
            }
        }
    } else {
        // 1D output - just concatenate
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
#[allow(clippy::needless_range_loop)] // Index used for multiple array accesses
fn compute_concatenate_indices(args: &[LinOp], axis: i64) -> Vec<i64> {
    if args.is_empty() {
        return vec![];
    }

    let mut indices = Vec::new();

    // Build flat index arrays for each arg
    let arg_offsets: Vec<i64> = args
        .iter()
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

    // Guard against axis out of bounds
    if axis >= n_dims {
        return arg_offsets;
    }

    // Build the result shape
    let mut result_shape = shapes[0].clone();
    result_shape[axis] = shapes
        .iter()
        .map(|s| s.get(axis).copied().unwrap_or(1))
        .sum();

    let total = result_shape.iter().product::<usize>();

    // Iterate over result indices in Fortran order
    let mut result_idx = vec![0usize; n_dims];
    for _ in 0..total {
        // Find which arg this belongs to
        let mut arg_idx = 0;
        let mut axis_offset = 0;
        for (i, shape) in shapes.iter().enumerate() {
            let shape_at_axis = shape.get(axis).copied().unwrap_or(1);
            if result_idx[axis] < axis_offset + shape_at_axis {
                arg_idx = i;
                break;
            }
            axis_offset += shape_at_axis;
        }

        // Compute flat index within the arg
        let mut local_idx = result_idx.clone();
        local_idx[axis] -= axis_offset;

        let mut flat = 0i64;
        let mut stride = 1i64;
        for dim in 0..n_dims {
            flat += local_idx[dim] as i64 * stride;
            stride *= shapes[arg_idx].get(dim).copied().unwrap_or(1) as i64;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linop::OpType;
    use crate::operations::process_reshape;
    use crate::tensor::CONSTANT_ID;
    use std::collections::HashMap;

    fn make_ctx(var_length: i64) -> ProcessingContext {
        let mut id_to_col = HashMap::new();
        id_to_col.insert(1, 0);

        let mut param_to_col = HashMap::new();
        param_to_col.insert(CONSTANT_ID, 0);

        let mut param_to_size = HashMap::new();
        param_to_size.insert(CONSTANT_ID, 1);

        ProcessingContext {
            id_to_col,
            param_to_col,
            param_to_size,
            var_length,
            param_size_plus_one: 1,
        }
    }

    #[test]
    fn test_transpose_2d() {
        let ctx = make_ctx(4);

        // Create variable (2x2)
        let var_op = LinOp {
            op_type: OpType::Variable,
            shape: vec![2, 2],
            args: vec![],
            data: LinOpData::Int(1),
        };

        // Transpose it - LinOpData::None triggers default 2D transpose
        let transpose_op = LinOp {
            op_type: OpType::Transpose,
            shape: vec![2, 2],
            args: vec![var_op],
            data: LinOpData::None,
        };

        let tensor = process_transpose(&transpose_op, &ctx);

        // Transpose of identity should permute rows
        assert_eq!(tensor.nnz(), 4);
        // Row order should be permuted for column-major transpose
    }

    #[test]
    fn test_reshape() {
        let ctx = make_ctx(4);

        // Create variable (2x2)
        let var_op = LinOp {
            op_type: OpType::Variable,
            shape: vec![2, 2],
            args: vec![],
            data: LinOpData::Int(1),
        };

        // Reshape to (4,)
        let reshape_op = LinOp {
            op_type: OpType::Reshape,
            shape: vec![4],
            args: vec![var_op],
            data: LinOpData::None,
        };

        let tensor = process_reshape(&reshape_op, &ctx);

        // Reshape doesn't change the data, just the shape
        assert_eq!(tensor.nnz(), 4);
        assert_eq!(tensor.shape.0, 4);
    }

    #[test]
    fn test_hstack() {
        // Two variables that will be horizontally stacked
        let mut id_to_col = HashMap::new();
        id_to_col.insert(1, 0);
        id_to_col.insert(2, 4);

        let mut param_to_col = HashMap::new();
        param_to_col.insert(CONSTANT_ID, 0);

        let mut param_to_size = HashMap::new();
        param_to_size.insert(CONSTANT_ID, 1);

        let ctx = ProcessingContext {
            id_to_col,
            param_to_col,
            param_to_size,
            var_length: 8,
            param_size_plus_one: 1,
        };

        let var1 = LinOp {
            op_type: OpType::Variable,
            shape: vec![2, 2],
            args: vec![],
            data: LinOpData::Int(1),
        };

        let var2 = LinOp {
            op_type: OpType::Variable,
            shape: vec![2, 2],
            args: vec![],
            data: LinOpData::Int(2),
        };

        let hstack_op = LinOp {
            op_type: OpType::Hstack,
            shape: vec![2, 4],
            args: vec![var1, var2],
            data: LinOpData::None,
        };

        let tensor = process_hstack(&hstack_op, &ctx);

        // Should have 8 non-zeros (4 from each variable)
        assert_eq!(tensor.nnz(), 8);
        assert_eq!(tensor.shape.0, 8); // 2*4 = 8 rows in flattened form
    }

    #[test]
    fn test_vstack() {
        let mut id_to_col = HashMap::new();
        id_to_col.insert(1, 0);
        id_to_col.insert(2, 4);

        let mut param_to_col = HashMap::new();
        param_to_col.insert(CONSTANT_ID, 0);

        let mut param_to_size = HashMap::new();
        param_to_size.insert(CONSTANT_ID, 1);

        let ctx = ProcessingContext {
            id_to_col,
            param_to_col,
            param_to_size,
            var_length: 8,
            param_size_plus_one: 1,
        };

        let var1 = LinOp {
            op_type: OpType::Variable,
            shape: vec![2, 2],
            args: vec![],
            data: LinOpData::Int(1),
        };

        let var2 = LinOp {
            op_type: OpType::Variable,
            shape: vec![2, 2],
            args: vec![],
            data: LinOpData::Int(2),
        };

        let vstack_op = LinOp {
            op_type: OpType::Vstack,
            shape: vec![4, 2],
            args: vec![var1, var2],
            data: LinOpData::None,
        };

        let tensor = process_vstack(&vstack_op, &ctx);

        // Should have 8 non-zeros
        assert_eq!(tensor.nnz(), 8);
        assert_eq!(tensor.shape.0, 8); // 4*2 = 8 rows
    }

    #[test]
    fn test_index_simple() {
        let ctx = make_ctx(4);

        // Create variable (2x2)
        let var_op = LinOp {
            op_type: OpType::Variable,
            shape: vec![2, 2],
            args: vec![],
            data: LinOpData::Int(1),
        };

        // Index [0:2, 0:1] - first column
        let index_op = LinOp {
            op_type: OpType::Index,
            shape: vec![2],
            args: vec![var_op],
            data: LinOpData::Slices(vec![
                SliceData {
                    start: 0,
                    stop: 2,
                    step: 1,
                },
                SliceData {
                    start: 0,
                    stop: 1,
                    step: 1,
                },
            ]),
        };

        let tensor = process_index(&index_op, &ctx);

        // Should select 2 elements (first column)
        assert_eq!(tensor.nnz(), 2);
    }

    #[test]
    fn test_promote() {
        let ctx = make_ctx(2);

        // Create 1D variable
        let var_op = LinOp {
            op_type: OpType::Variable,
            shape: vec![2],
            args: vec![],
            data: LinOpData::Int(1),
        };

        // Promote to 2D
        let promote_op = LinOp {
            op_type: OpType::Promote,
            shape: vec![2, 1],
            args: vec![var_op],
            data: LinOpData::None,
        };

        let tensor = process_promote(&promote_op, &ctx);

        // Should still have same non-zeros
        assert_eq!(tensor.nnz(), 2);
    }
}
