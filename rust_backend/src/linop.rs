use numpy::{ndarray::Array2, PyReadonlyArray0, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use std::borrow::Borrow;

use crate::view::View;
use crate::NdArray;

// TinyVec for n-dimensional?
#[derive(Debug)]
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

#[derive(Debug)]
pub(crate) struct Linop<'a> {
    pub(crate) shape: CvxpyShape,
    pub(crate) kind: LinopKind<'a>,
}

#[derive(Debug)]
pub(crate) enum LinopKind<'a> {
    Variable(i64),
    Mul {
        lhs: Box<Linop<'a>>,
        rhs: Box<Linop<'a>>,
    },
    RMul {
        lhs: Box<Linop<'a>>,
        rhs: Box<Linop<'a>>,
    },
    MulElem {
        lhs: Box<Linop<'a>>,
        rhs: Box<Linop<'a>>,
    },
    Sum(Vec<Linop<'a>>),
    Neg(Box<Linop<'a>>),
    Transpose(Box<Linop<'a>>),
    SumEntries(Box<Linop<'a>>),
    ScalarConst(f64),
    DenseConst(Array2<f64>),
    SparseConst(&'a crate::SparseMatrix),
    Param(i64),
    Reshape(Box<Linop<'a>>),
    Promote(Box<Linop<'a>>),
    // Hstack(Vec<Linop<'a>>),
    // Vstack(Vec<Linop<'a>>),
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
            "sum" => LinopKind::Sum(Vec::extract_bound(&ob.getattr(intern!(ob.py(), "args"))?)?),
            "mul" => {
                let lhs = Linop::extract_bound(&ob.getattr(intern!(ob.py(), "data"))?)?;
                let rhs = {
                    let args = Vec::extract_bound(&ob.getattr(intern!(ob.py(), "args"))?)?;
                    let arg = args
                        .first()
                        .ok_or(PyValueError::new_err("Empty args list"))?;
                    Linop::extract_bound(arg)?
                };
                LinopKind::Mul {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }
            }
            "neg" => {
                let args = Vec::extract_bound(&ob.getattr(intern!(ob.py(), "args"))?)?;
                let arg = args
                    .first()
                    .ok_or(PyValueError::new_err("Empty args list"))?;
                LinopKind::Neg(Box::new(Linop::extract_bound(arg)?))
            }
            "promote" => {
                let args = Vec::extract_bound(&ob.getattr(intern!(ob.py(), "args"))?)?;
                let arg = args
                    .first()
                    .ok_or(PyValueError::new_err("Empty args list"))?;
                LinopKind::Promote(Box::new(Linop::extract_bound(arg)?))
            }
            "transpose" => {
                let args = Vec::extract_bound(&ob.getattr(intern!(ob.py(), "args"))?)?;
                let arg = args
                    .first()
                    .ok_or(PyValueError::new_err("Empty args list"))?;
                LinopKind::Transpose(Box::new(Linop::extract_bound(arg)?))
            }
            "reshape" => {
                let args = Vec::extract_bound(&ob.getattr(intern!(ob.py(), "args"))?)?;
                let arg = args
                    .first()
                    .ok_or(PyValueError::new_err("Empty args list"))?;
                LinopKind::Reshape(Box::new(Linop::extract_bound(arg)?))
            }
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
        Ok(Linop { shape, kind })
    }
}
