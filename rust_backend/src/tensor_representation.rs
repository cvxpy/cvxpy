use std::cmp::PartialEq;

/// Sparse representation of a 3D Tensor, semantically similar to COO format,
/// with one extra dimension. Here, 'row' is axis 0, 'col' axis 1, and 'parameter_offset' axis 2.
#[derive(Debug, PartialEq)]
pub(crate) struct TensorRepresentation {
    pub(crate) data: Vec<f64>,
    pub(crate) row: Vec<u64>,
    pub(crate) col: Vec<u64>,
    pub(crate) parameter_offset: Vec<u64>,
}

impl TensorRepresentation {
    /// Concatenates the row, col, parameter_offset, and data fields of a list of TensorRepresentations.
    pub fn combine(tensors: Vec<TensorRepresentation>) -> TensorRepresentation {
        let mut data = Vec::new();
        let mut row = Vec::new();
        let mut col = Vec::new();
        let mut parameter_offset = Vec::new();

        for t in tensors {
            data.extend_from_slice(t.data.as_slice());
            row.extend_from_slice(t.row.as_slice());
            col.extend_from_slice(t.col.as_slice());
            parameter_offset.extend_from_slice(t.parameter_offset.as_slice());
        }

        TensorRepresentation {
            data,
            row,
            col,
            parameter_offset,
        }
    }

    /// Returns true if the given TensorRepresentation is equal to self.
    fn eq(&self, other: &TensorRepresentation) -> bool {
        self.data == other.data
            && self.row == other.row
            && self.col == other.col
            && self.parameter_offset == other.parameter_offset
    }
}
