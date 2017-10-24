#!/usr/bin/env python
"""matdata.py

Class definition of MatData
This class is used for interface of scipy and C API.
This is the module of sdpa-p.

December 2010, Kenta KATO
"""

from scipy import sparse, array

class MatData(object):
    """Description of a sparse matrix
    Attributes:
      size_row: Size of row
      size_col: Size of column
      values: List of Values
      rowind: List of row indices
      colptr: List of start index of each column
    """
    def __init__(self, mat=None,
                 values=None, rowind=None, colptr=None, size=None):
        """Constructor for sparse matrix
        Args:
          mat: input sparse matrix
          values: List of Values
          rowind: List of row indices
          colptr: List of start index of each column
          size: Tuple (size_row, size_col)
        """
        if sparse.issparse(mat):
            mat2 = mat.tocsc() if not sparse.isspmatrix_csc(mat) else mat
            self.size_row, self.size_col = mat2.shape
            self.values = list(mat2.data)
            self.rowind = list(mat2.indices)
            self.colptr = list(mat2.indptr)
        elif (isinstance(values, list) and isinstance(rowind, list) and
              isinstance(colptr, list) and isinstance(size, tuple)):
            if (len(values) != len(rowind) or len(colptr) != size[1] + 1):
                raise ValueError('size of val, row, col list must be same')
            self.size_row, self.size_col = size
            self.values, self.rowind, self.colptr = values, rowind, colptr
        else:
            raise TypeError('Input arg must be sparse matrix or lists.')

    def __len__(self):
        """Return num of nonzero elements in matrix"""
        return len(self.values)

    def size(self):
        """Return the size of matrix"""
        return (self.size_row, self.size_col)


