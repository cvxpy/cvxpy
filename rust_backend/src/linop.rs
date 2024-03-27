use pyo3::intern;
use pyo3::prelude::*;
use std::borrow::Borrow;

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
}

// shape: CvxpyShape
// Merge into kind:
//      type: str
//      data: None | int | ndarray | TinyVec[PySlice]
//      args: Vec<Linop>
pub(crate) struct Linop {
    pub(crate) shape: CvxpyShape,
    pub(crate) kind: LinopKind,
}

type NdArray = ();

pub(crate) enum LinopKind {
    Variable(i64),
    Mul { lhs: Box<Linop>, rhs: Box<Linop> },
    Rmul { lhs: Box<Linop>, rhs: Box<Linop> },
    MulElem { lhs: Box<Linop>, rhs: Box<Linop> },
    Sum,
    Neg,
    Transpose,
    SumEntries,
    ScalarConst(f64),
    DenseConst(NdArray),
    SparseConst(crate::SparseMatrix),
    Param(i64),
}

impl<'py> FromPyObject<'py> for CvxpyShape {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        todo!();
    }
}

impl<'py> FromPyObject<'py> for Linop {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let pylinop = ob.borrow();
        let shape: CvxpyShape = pylinop.getattr(intern!(ob.py(), "shape"))?.extract()?;
        todo!();
    }
}
