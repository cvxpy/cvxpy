use numpy::{ndarray::Array2, PyReadonlyArray0, PyReadonlyArray2};
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
    pub(crate) args: Vec<Linop<'a>>,
}

pub(crate) enum LinopKind<'a> {
    Variable(i64),
    Mul { lhs: Box<Linop<'a>> },
    Rmul { rhs: Box<Linop<'a>> },
    MulElem { lhs: Box<Linop<'a>> },
    Sum,
    Neg,
    Transpose,
    SumEntries,
    ScalarConst(f64),
    DenseConst(Array2<f64>),
    SparseConst(&'a crate::SparseMatrix),
    Param(i64),
    Reshape,
    Promote,
}

impl<'py> FromPyObject<'py> for CvxpyShape {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let v = <Vec<u64> as FromPyObject>::extract_bound(ob)?;
        Ok(match *v {
            [] => CvxpyShape::D0,
            [n] => CvxpyShape::D1(n),
            [m, n] => CvxpyShape::D2(m, n),
            _ => panic!("Only support 2D expressions"),
        })
    }
}

impl<'py> FromPyObject<'py> for Linop<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let shape: CvxpyShape = ob.getattr(intern!(ob.py(), "shape"))?.extract()?;
        let type_string = ob.getattr(intern!(ob.py(), "type"))?;
        let kind: LinopKind<'py> = match type_string.extract()? {
            "sum" => LinopKind::Sum,
            "mul" => {
                let lhs = Linop::extract_bound(&ob.getattr(intern!(ob.py(), "data"))?)?;
                LinopKind::Mul { lhs: Box::new(lhs) }
            }
            "neg" => LinopKind::Neg,
            "promote" => LinopKind::Promote,
            "transpose" => LinopKind::Transpose,
            "reshape" => LinopKind::Reshape,
            "variable" => {
                LinopKind::Variable(i64::extract_bound(&ob.getattr(intern!(ob.py(), "data"))?)?)
            }
            "sparse_const" => {
                todo!()
            }
            "dense_const" => {
                let data = ob.getattr(intern!(ob.py(), "data"))?;
                match shape {
                    CvxpyShape::D2(_, _) => {
                        let array = PyReadonlyArray2::extract_bound(&data)?;
                        LinopKind::DenseConst(array.as_array().to_owned())
                    }
                    CvxpyShape::D1(_) => panic!("Should be no 1D arrays"),
                    CvxpyShape::D0 => {
                        let array = PyReadonlyArray0::extract_bound(&data)?;
                        let f = *array.get(()).unwrap();
                        LinopKind::ScalarConst(f)
                    }
                }
            }
            "scalar_const" => {
                let f = f64::extract_bound(&ob.getattr(intern!(ob.py(), "data"))?)?;
                LinopKind::ScalarConst(f)
            }
            _ => {
                todo!()
            }
        };
        let args = Vec::extract_bound(&ob.getattr(intern!(ob.py(), "args"))?)?;
        Ok(Linop { shape, kind, args })
    }
}
