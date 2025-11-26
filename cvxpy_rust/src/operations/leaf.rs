//! Leaf node operations: variable, const, param
//!
//! These operations create tensors for the leaf nodes of the expression tree.

use crate::linop::{LinOp, LinOpData};
use crate::tensor::{SparseTensor, SparseTensorBuilder, CONSTANT_ID};
use super::ProcessingContext;

/// Process a variable node
///
/// Creates an identity matrix of size n, placed at the variable's column offset.
/// The tensor represents the coefficient matrix where each variable component
/// maps to itself with coefficient 1.
pub fn process_variable(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    let n = lin_op.size();

    // Get variable ID from data
    let var_id = match &lin_op.data {
        LinOpData::Int(id) => *id,
        _ => panic!("Variable node must have integer data (variable ID)"),
    };

    // Get column offset for this variable
    let col_offset = ctx.var_col(var_id);

    // Create identity matrix: I_n at position (0:n, col_offset:col_offset+n)
    // Parameter offset is the constant slice (non-parametric)
    let param_offset = ctx.const_param();

    let mut builder = SparseTensorBuilder::new(
        (n, ctx.var_length as usize + 1),
        n,
    );
    builder.add_variable_identity(n, col_offset, param_offset);
    builder.build()
}

/// Process a scalar constant node
///
/// Creates a single-element tensor with the scalar value.
pub fn process_scalar_const(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    let value = match &lin_op.data {
        LinOpData::Float(v) => *v,
        LinOpData::Int(v) => *v as f64,
        _ => panic!("Scalar const node must have numeric data"),
    };

    if value == 0.0 {
        return SparseTensor::empty((1, ctx.var_length as usize + 1));
    }

    // Place in the constant column (var_length)
    let col_offset = ctx.const_col();
    let param_offset = ctx.const_param();

    let mut tensor = SparseTensor::with_capacity(
        (1, ctx.var_length as usize + 1),
        1,
    );
    tensor.push(value, 0, col_offset, param_offset);
    tensor
}

/// Process a dense constant node
///
/// Creates a column vector tensor from the dense array data.
pub fn process_dense_const(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    let (data, shape) = match &lin_op.data {
        LinOpData::DenseArray { data, shape } => (data, shape),
        _ => panic!("Dense const node must have dense array data"),
    };

    let n = lin_op.size();
    let col_offset = ctx.const_col();
    let param_offset = ctx.const_param();

    // Count non-zeros for capacity estimation
    let nnz = data.iter().filter(|&&x| x != 0.0).count();

    let mut tensor = SparseTensor::with_capacity(
        (n, ctx.var_length as usize + 1),
        nnz,
    );

    // Flatten data in Fortran (column-major) order for output
    // Input data from NumPy is in row-major (C) order
    // For a 2D array with shape (m, n):
    //   - NumPy row-major: element at (i, j) is at index i * n + j
    //   - Fortran col-major output: element at (i, j) goes to row i + j * m
    if shape.len() == 2 {
        let (m, n_cols) = (shape[0], shape[1]);
        for j in 0..n_cols {
            for i in 0..m {
                let input_idx = i * n_cols + j;  // NumPy row-major index
                let output_row = j * m + i;      // Fortran column-major row
                let value = data[input_idx];
                if value != 0.0 {
                    tensor.push(value, output_row as i64, col_offset, param_offset);
                }
            }
        }
    } else if shape.len() == 1 {
        // 1D array
        for (i, &value) in data.iter().enumerate() {
            if value != 0.0 {
                tensor.push(value, i as i64, col_offset, param_offset);
            }
        }
    } else if shape.is_empty() {
        // Scalar
        if !data.is_empty() && data[0] != 0.0 {
            tensor.push(data[0], 0, col_offset, param_offset);
        }
    } else {
        // Higher dimensional - convert from row-major (C) to column-major (Fortran)
        // General formula for n-dimensional arrays:
        // C-order index -> coordinates -> F-order index

        let total_size: usize = shape.iter().product();
        let n_dims = shape.len();

        // Precompute strides for C and F order
        let mut c_strides = vec![1usize; n_dims];
        let mut f_strides = vec![1usize; n_dims];
        for i in (0..n_dims - 1).rev() {
            c_strides[i] = c_strides[i + 1] * shape[i + 1];
        }
        for i in 1..n_dims {
            f_strides[i] = f_strides[i - 1] * shape[i - 1];
        }

        for c_idx in 0..total_size {
            let value = data[c_idx];
            if value != 0.0 {
                // Convert C index to coordinates
                let mut coords = vec![0usize; n_dims];
                let mut remaining = c_idx;
                for d in 0..n_dims {
                    coords[d] = remaining / c_strides[d];
                    remaining %= c_strides[d];
                }

                // Convert coordinates to F index
                let f_idx: usize = coords.iter()
                    .zip(f_strides.iter())
                    .map(|(&c, &s)| c * s)
                    .sum();

                tensor.push(value, f_idx as i64, col_offset, param_offset);
            }
        }
    }

    tensor
}

/// Process a sparse constant node
///
/// Creates a column vector tensor from the sparse array data (CSC format).
pub fn process_sparse_const(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    let (data, indices, indptr, shape) = match &lin_op.data {
        LinOpData::SparseArray { data, indices, indptr, shape } => {
            (data, indices, indptr, shape)
        }
        _ => panic!("Sparse const node must have sparse array data"),
    };

    let n = lin_op.size();
    let col_offset = ctx.const_col();
    let param_offset = ctx.const_param();

    let mut tensor = SparseTensor::with_capacity(
        (n, ctx.var_length as usize + 1),
        data.len(),
    );

    // CSC format: iterate over columns
    let n_cols = indptr.len() - 1;
    let m = shape.0;

    for j in 0..n_cols {
        let start = indptr[j] as usize;
        let end = indptr[j + 1] as usize;
        for idx in start..end {
            let i = indices[idx] as usize;
            let value = data[idx];
            if value != 0.0 {
                // Convert (i, j) to flat index in Fortran order
                let flat_idx = (j * m + i) as i64;
                tensor.push(value, flat_idx, col_offset, param_offset);
            }
        }
    }

    tensor
}

/// Process a parameter node
///
/// Creates a tensor where each parameter element maps to its own parameter slice.
/// The tensor has an identity-like structure across the parameter axis.
pub fn process_param(lin_op: &LinOp, ctx: &ProcessingContext) -> SparseTensor {
    let n = lin_op.size();

    // Get parameter ID from data
    let param_id = match &lin_op.data {
        LinOpData::Int(id) => *id,
        _ => panic!("Param node must have integer data (parameter ID)"),
    };

    // Get parameter column offset and size
    let param_col_offset = ctx.param_col(param_id);
    let param_size = ctx.param_size(param_id) as usize;

    // Place in the constant column (the 'b' column)
    let col_offset = ctx.const_col();

    // For parameters, we create n entries where:
    // - Each row i maps to parameter slice (param_col_offset + i % param_size)
    // - This creates a diagonal structure across the parameter dimension
    let mut tensor = SparseTensor::with_capacity(
        (n, ctx.var_length as usize + 1),
        n,
    );

    for i in 0..n {
        let param_offset = if param_size == 0 {
            param_col_offset
        } else {
            param_col_offset + (i % param_size) as i64
        };
        tensor.push(1.0, i as i64, col_offset, param_offset);
    }

    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linop::OpType;
    use std::collections::HashMap;

    fn make_ctx() -> ProcessingContext {
        let mut id_to_col = HashMap::new();
        id_to_col.insert(0, 0);  // Variable 0 at column 0
        id_to_col.insert(1, 5);  // Variable 1 at column 5

        let mut param_to_col = HashMap::new();
        param_to_col.insert(0, 0);  // Parameter 0 at slice 0
        param_to_col.insert(CONSTANT_ID, 2);  // Constant at slice 2

        let mut param_to_size = HashMap::new();
        param_to_size.insert(0, 2);  // Parameter 0 has size 2
        param_to_size.insert(CONSTANT_ID, 1);

        ProcessingContext {
            id_to_col,
            param_to_col,
            param_to_size,
            var_length: 10,
            param_size_plus_one: 3,
        }
    }

    #[test]
    fn test_variable() {
        let ctx = make_ctx();
        let lin_op = LinOp {
            op_type: OpType::Variable,
            shape: vec![3],
            args: vec![],
            data: LinOpData::Int(0),  // Variable ID 0
        };

        let tensor = process_variable(&lin_op, &ctx);

        assert_eq!(tensor.nnz(), 3);
        assert_eq!(tensor.data, vec![1.0, 1.0, 1.0]);
        assert_eq!(tensor.rows, vec![0, 1, 2]);
        assert_eq!(tensor.cols, vec![0, 1, 2]);  // Column 0, 1, 2 for variable 0
    }

    #[test]
    fn test_scalar_const() {
        let ctx = make_ctx();
        let lin_op = LinOp {
            op_type: OpType::ScalarConst,
            shape: vec![],
            args: vec![],
            data: LinOpData::Float(3.14),
        };

        let tensor = process_scalar_const(&lin_op, &ctx);

        assert_eq!(tensor.nnz(), 1);
        assert_eq!(tensor.data, vec![3.14]);
        assert_eq!(tensor.cols[0], 10);  // Constant column (var_length)
    }

    #[test]
    fn test_dense_const() {
        let ctx = make_ctx();
        let lin_op = LinOp {
            op_type: OpType::DenseConst,
            shape: vec![3],
            args: vec![],
            data: LinOpData::DenseArray {
                data: vec![1.0, 0.0, 2.0],
                shape: vec![3],
            },
        };

        let tensor = process_dense_const(&lin_op, &ctx);

        assert_eq!(tensor.nnz(), 2);  // Two non-zeros
        assert_eq!(tensor.data, vec![1.0, 2.0]);
        assert_eq!(tensor.rows, vec![0, 2]);
    }

    #[test]
    fn test_dense_const_3d() {
        // Test n-dimensional (>2D) constant conversion from C to F order
        let ctx = make_ctx();

        // 3D array shape (2, 3, 4) with single non-zero at position [0, 0, 1]
        // In C-order, [0,0,1] is at index: 0*3*4 + 0*4 + 1 = 1
        // In F-order, [0,0,1] is at index: 0 + 0*2 + 1*2*3 = 6
        let mut data = vec![0.0; 24];
        data[1] = 42.0;  // C-order index for [0,0,1]

        let lin_op = LinOp {
            op_type: OpType::DenseConst,
            shape: vec![2, 3, 4],
            args: vec![],
            data: LinOpData::DenseArray {
                data,
                shape: vec![2, 3, 4],
            },
        };

        let tensor = process_dense_const(&lin_op, &ctx);

        assert_eq!(tensor.nnz(), 1);
        assert_eq!(tensor.data, vec![42.0]);
        assert_eq!(tensor.rows, vec![6]);  // F-order index
    }
}
