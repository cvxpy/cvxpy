//! Sparse tensor representation in COO format
//!
//! This module provides the SparseTensor type which stores 3D tensor data
//! in coordinate (COO) format, matching CVXPY's TensorRepresentation.

use std::collections::HashMap;

/// Constant ID used for non-parametric entries
pub const CONSTANT_ID: i64 = -1;

/// Sparse tensor in COO format
///
/// Represents a 3D tensor with dimensions (rows, cols, param_slices).
/// This matches Python's TensorRepresentation from canon_backend.py.
#[derive(Debug, Clone)]
pub struct SparseTensor {
    pub data: Vec<f64>,
    pub rows: Vec<i64>,
    pub cols: Vec<i64>,
    pub param_offsets: Vec<i64>,
    pub shape: (usize, usize),
}

impl SparseTensor {
    /// Create an empty tensor with given shape
    pub fn empty(shape: (usize, usize)) -> Self {
        SparseTensor {
            data: Vec::new(),
            rows: Vec::new(),
            cols: Vec::new(),
            param_offsets: Vec::new(),
            shape,
        }
    }

    /// Create an empty tensor with pre-allocated capacity
    pub fn with_capacity(shape: (usize, usize), capacity: usize) -> Self {
        SparseTensor {
            data: Vec::with_capacity(capacity),
            rows: Vec::with_capacity(capacity),
            cols: Vec::with_capacity(capacity),
            param_offsets: Vec::with_capacity(capacity),
            shape,
        }
    }

    /// Number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Check if the tensor is empty
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Add a single entry to the tensor
    #[inline]
    pub fn push(&mut self, value: f64, row: i64, col: i64, param_offset: i64) {
        self.data.push(value);
        self.rows.push(row);
        self.cols.push(col);
        self.param_offsets.push(param_offset);
    }

    /// Extend this tensor with entries from another tensor
    pub fn extend(&mut self, other: SparseTensor) {
        self.data.extend(other.data);
        self.rows.extend(other.rows);
        self.cols.extend(other.cols);
        self.param_offsets.extend(other.param_offsets);
    }

    /// Negate all data values in place
    pub fn negate_in_place(&mut self) {
        for d in &mut self.data {
            *d = -*d;
        }
    }

    /// Scale all data values in place
    pub fn scale_in_place(&mut self, factor: f64) {
        for d in &mut self.data {
            *d *= factor;
        }
    }

    /// Offset all row indices in place
    pub fn offset_rows_in_place(&mut self, offset: i64) {
        for r in &mut self.rows {
            *r += offset;
        }
    }

    /// Offset all column indices in place
    #[allow(dead_code)]
    pub fn offset_cols_in_place(&mut self, offset: i64) {
        for c in &mut self.cols {
            *c += offset;
        }
    }

    /// Offset all parameter indices in place
    #[allow(dead_code)]
    pub fn offset_params_in_place(&mut self, offset: i64) {
        for p in &mut self.param_offsets {
            *p += offset;
        }
    }

    /// Select rows by index array (creates new tensor)
    /// OPTIMIZATION: Uses fast paths for common patterns
    pub fn select_rows(&self, row_indices: &[i64]) -> SparseTensor {
        // Fast path 1: Empty input
        if row_indices.is_empty() {
            return SparseTensor::empty((0, self.shape.1));
        }

        // Fast path 2: Identity permutation (no change needed)
        if self.is_identity_permutation(row_indices) {
            return self.clone();
        }

        // Fast path 3: Simple offset (contiguous range starting from offset)
        if let Some(offset) = self.check_contiguous_with_offset(row_indices) {
            return self.select_contiguous_rows(offset, row_indices.len());
        }

        // Fast path 4: Reversed identity permutation
        if self.is_reversed_identity(row_indices) {
            return self.reverse_rows();
        }

        // General case: use HashMap
        self.select_rows_general(row_indices)
    }

    /// Check if row_indices is an identity permutation [0, 1, 2, ..., n-1]
    #[inline]
    fn is_identity_permutation(&self, row_indices: &[i64]) -> bool {
        if row_indices.len() != self.shape.0 {
            return false;
        }
        row_indices.iter().enumerate().all(|(i, &r)| r == i as i64)
    }

    /// Check if row_indices is a contiguous range with offset [offset, offset+1, ..., offset+n-1]
    /// Returns the offset if so
    #[inline]
    fn check_contiguous_with_offset(&self, row_indices: &[i64]) -> Option<i64> {
        if row_indices.is_empty() {
            return Some(0);
        }
        let offset = row_indices[0];
        if row_indices
            .iter()
            .enumerate()
            .all(|(i, &r)| r == offset + i as i64)
        {
            Some(offset)
        } else {
            None
        }
    }

    /// Check if row_indices is reversed identity [n-1, n-2, ..., 1, 0]
    #[inline]
    fn is_reversed_identity(&self, row_indices: &[i64]) -> bool {
        if row_indices.len() != self.shape.0 {
            return false;
        }
        let n = row_indices.len();
        row_indices
            .iter()
            .enumerate()
            .all(|(i, &r)| r == (n - 1 - i) as i64)
    }

    /// Select contiguous rows starting from offset
    fn select_contiguous_rows(&self, offset: i64, count: usize) -> SparseTensor {
        let end_row = offset + count as i64;

        // Count entries in range for capacity estimation
        let est_nnz = self
            .rows
            .iter()
            .filter(|&&r| r >= offset && r < end_row)
            .count();

        let mut result = SparseTensor::with_capacity((count, self.shape.1), est_nnz);

        for i in 0..self.nnz() {
            let row = self.rows[i];
            if row >= offset && row < end_row {
                result.push(
                    self.data[i],
                    row - offset, // Adjust row index
                    self.cols[i],
                    self.param_offsets[i],
                );
            }
        }

        result
    }

    /// Reverse all row indices
    fn reverse_rows(&self) -> SparseTensor {
        let n_rows = self.shape.0 as i64;
        let mut result = SparseTensor::with_capacity(self.shape, self.nnz());

        for i in 0..self.nnz() {
            result.push(
                self.data[i],
                n_rows - 1 - self.rows[i], // Reverse row index
                self.cols[i],
                self.param_offsets[i],
            );
        }

        result
    }

    /// General row selection using HashMap (fallback)
    fn select_rows_general(&self, row_indices: &[i64]) -> SparseTensor {
        // Build mapping from old row to new positions
        let mut row_map: HashMap<i64, Vec<usize>> = HashMap::with_capacity(row_indices.len());
        for (new_idx, &old_row) in row_indices.iter().enumerate() {
            row_map.entry(old_row).or_default().push(new_idx);
        }

        // Estimate capacity
        let mut result = SparseTensor::with_capacity(
            (row_indices.len(), self.shape.1),
            self.nnz() * row_indices.len() / self.shape.0.max(1),
        );

        // Select entries
        for i in 0..self.nnz() {
            if let Some(new_positions) = row_map.get(&self.rows[i]) {
                for &new_row in new_positions {
                    result.push(
                        self.data[i],
                        new_row as i64,
                        self.cols[i],
                        self.param_offsets[i],
                    );
                }
            }
        }

        result
    }

    /// Combine multiple tensors into one (concatenate all entries)
    pub fn combine(tensors: Vec<SparseTensor>) -> SparseTensor {
        if tensors.is_empty() {
            return SparseTensor::empty((0, 0));
        }

        let total_nnz: usize = tensors.iter().map(|t| t.nnz()).sum();
        let shape = tensors[0].shape;

        let mut result = SparseTensor::with_capacity(shape, total_nnz);
        for tensor in tensors {
            result.extend(tensor);
        }
        result
    }
}

/// Builder for SparseTensor with efficient accumulation
#[derive(Debug)]
pub struct SparseTensorBuilder {
    data: Vec<f64>,
    rows: Vec<i64>,
    cols: Vec<i64>,
    param_offsets: Vec<i64>,
    shape: (usize, usize),
}

impl SparseTensorBuilder {
    /// Create a new builder with given shape and capacity
    pub fn new(shape: (usize, usize), capacity: usize) -> Self {
        SparseTensorBuilder {
            data: Vec::with_capacity(capacity),
            rows: Vec::with_capacity(capacity),
            cols: Vec::with_capacity(capacity),
            param_offsets: Vec::with_capacity(capacity),
            shape,
        }
    }

    /// Add a single entry
    #[inline]
    pub fn push(&mut self, value: f64, row: i64, col: i64, param_offset: i64) {
        self.data.push(value);
        self.rows.push(row);
        self.cols.push(col);
        self.param_offsets.push(param_offset);
    }

    /// Add an identity matrix block for a variable
    pub fn add_variable_identity(&mut self, size: usize, col_offset: i64, param_offset: i64) {
        for i in 0..size {
            self.push(1.0, i as i64, col_offset + i as i64, param_offset);
        }
    }

    /// Add a constant column vector
    #[allow(dead_code)]
    pub fn add_constant_column(&mut self, data: &[f64], col_offset: i64, param_offset: i64) {
        for (i, &value) in data.iter().enumerate() {
            if value != 0.0 {
                self.push(value, i as i64, col_offset, param_offset);
            }
        }
    }

    /// Add sparse CSC data
    #[allow(dead_code)]
    pub fn add_sparse_csc(
        &mut self,
        values: &[f64],
        indices: &[i64],
        indptr: &[i64],
        _shape: (usize, usize),
        row_offset: i64,
        col_offset: i64,
        param_offset: i64,
    ) {
        let n_cols = indptr.len() - 1;
        for j in 0..n_cols {
            let start = indptr[j] as usize;
            let end = indptr[j + 1] as usize;
            for idx in start..end {
                let row = indices[idx];
                let value = values[idx];
                if value != 0.0 {
                    self.push(value, row + row_offset, j as i64 + col_offset, param_offset);
                }
            }
        }
    }

    /// Build the final SparseTensor
    pub fn build(self) -> SparseTensor {
        SparseTensor {
            data: self.data,
            rows: self.rows,
            cols: self.cols,
            param_offsets: self.param_offsets,
            shape: self.shape,
        }
    }
}

/// Result structure returned to Python
#[derive(Debug)]
pub struct BuildMatrixResult {
    pub data: Vec<f64>,
    pub rows: Vec<i64>,
    pub cols: Vec<i64>,
    pub shape: (usize, usize),
}

impl BuildMatrixResult {
    /// Create from a SparseTensor by flattening the 3D structure to 2D
    ///
    /// The output matrix has shape (total_rows * (var_length + 1), param_size_plus_one)
    /// where the tensor is flattened in column-major (Fortran) order.
    pub fn from_tensor(tensor: SparseTensor, num_param_slices: usize) -> Self {
        let (n_rows, n_cols) = tensor.shape;
        let output_rows = n_rows * n_cols;
        let output_cols = num_param_slices;

        // Convert 3D COO to 2D COO
        // Output row = col * n_rows + row (column-major flattening)
        let flat_rows: Vec<i64> = tensor
            .rows
            .iter()
            .zip(tensor.cols.iter())
            .map(|(&r, &c)| c * (n_rows as i64) + r)
            .collect();

        BuildMatrixResult {
            data: tensor.data,
            rows: flat_rows,
            cols: tensor.param_offsets,
            shape: (output_rows, output_cols),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_tensor_basic() {
        let mut tensor = SparseTensor::empty((3, 4));
        tensor.push(1.0, 0, 0, 0);
        tensor.push(2.0, 1, 1, 0);
        tensor.push(3.0, 2, 2, 0);

        assert_eq!(tensor.nnz(), 3);
        assert_eq!(tensor.shape, (3, 4));
    }

    #[test]
    fn test_sparse_tensor_negate() {
        let mut tensor = SparseTensor::empty((2, 2));
        tensor.push(1.0, 0, 0, 0);
        tensor.push(-2.0, 1, 1, 0);

        tensor.negate_in_place();

        assert_eq!(tensor.data, vec![-1.0, 2.0]);
    }

    #[test]
    fn test_builder() {
        let mut builder = SparseTensorBuilder::new((3, 3), 10);
        builder.add_variable_identity(3, 0, 0);

        let tensor = builder.build();
        assert_eq!(tensor.nnz(), 3);
        assert_eq!(tensor.data, vec![1.0, 1.0, 1.0]);
        assert_eq!(tensor.rows, vec![0, 1, 2]);
        assert_eq!(tensor.cols, vec![0, 1, 2]);
    }
}
