use ndarray::ArrayView2;
use pyo3::intern;
use pyo3::prelude::*;
use std::borrow::Borrow;

use crate::NdArray;

// TinyVec for n-dimensional?
pub(crate) enum CvxpyShape {
    D0,
    D1(u64),
    D2(u64, u64),
}

impl CvxpyShape {
    pub fn numel(&self) -> u64 {
        use CvxpyShape::*;
        match *self {
            D0 => 1,
            D1(n) => n,
            D2(m, n) => m * n,
        }
    }

    pub fn broadcasted_shape(&self) -> (u64, u64) {
        use CvxpyShape::*;
        match *self {
            D0 => (1, 1),
            D1(n) => (1, n),
            D2(m, n) => (m, n),
        }
    }
}

// shape: CvxpyShape
// Merge into kind:
//      type: str
//      data: None | int | ndarray | TinyVec[PySlice]
//      args: Vec<Linop>

pub(crate) struct Linop<'a> {
    pub(crate) shape: CvxpyShape,
    pub(crate) kind: LinopKind<'a>,
}

pub(crate) enum LinopKind<'a> {
    Variable(i64),
    Mul { lhs: Box<Linop<'a>>, rhs: Box<Linop<'a>> },
    Rmul { lhs: Box<Linop<'a>>, rhs: Box<Linop<'a>> },
    MulElem { lhs: Box<Linop<'a>>, rhs: Box<Linop<'a>> },
    Sum,
    Neg,
    Transpose,
    SumEntries,
    ScalarConst(f64),
    DenseConst(ArrayView2<'a, f64>),
    SparseConst(&'a crate::SparseMatrix),
    Param(i64),
    Reshape,
    Promote,
}

impl<'py> FromPyObject<'py> for CvxpyShape {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        todo!();
    }
}

impl<'py> FromPyObject<'py> for Linop<'_> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let pylinop = ob.borrow();
        let shape: CvxpyShape = pylinop.getattr(intern!(ob.py(), "shape"))?.extract()?;
        todo!();
    }
}
