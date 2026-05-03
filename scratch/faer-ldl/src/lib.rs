//! Minimal Python wrapper around faer's sparse symmetric Bunch-Kaufman
//! (intranode LBL^T) factorization.
//!
//! Surface:
//!   ldl = faer_ldl.SparseLDL(n, indptr, indices, data)
//!     -- input is the UPPER triangle of an n-by-n symmetric matrix in
//!        scipy CSC layout (col_ptr, row_idx, values).
//!   x = ldl.solve(b)
//!   ldl.n, ldl.nnz_l
//!
//! Under the hood we force the supernodal path so that the factorization
//! actually does Bunch-Kaufman pivoting (the simplicial path silently
//! degrades to plain LDLT, no pivoting -- which is exactly the qdldl
//! failure mode we're trying to avoid).

use dyn_stack::{MemBuffer, MemStack};
use faer::{Conj, MatMut, Par, Side, Spec};
use faer::perm::PermRef;
use faer::linalg::cholesky::lblt::factor::LbltParams;
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, IntranodeLbltRef,
    SymbolicCholesky, SymmetricOrdering,
};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
struct SparseLDL {
    n: usize,
    symbolic: SymbolicCholesky<usize>,
    l_values: Vec<f64>,
    subdiag: Vec<f64>,
    perm_fwd: Vec<usize>,
    perm_inv: Vec<usize>,
}

fn err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

#[pymethods]
impl SparseLDL {
    #[new]
    fn new(
        n: usize,
        indptr: PyReadonlyArray1<'_, i64>,
        indices: PyReadonlyArray1<'_, i64>,
        data: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let indptr_i64 = indptr.as_slice().map_err(|e| err(format!("indptr: {e}")))?;
        let indices_i64 = indices.as_slice().map_err(|e| err(format!("indices: {e}")))?;
        let values = data.as_slice().map_err(|e| err(format!("data: {e}")))?;

        if indptr_i64.len() != n + 1 {
            return Err(err(format!(
                "indptr length {} != n+1 = {}",
                indptr_i64.len(),
                n + 1
            )));
        }
        if indices_i64.len() != values.len() {
            return Err(err(format!(
                "indices length {} != data length {}",
                indices_i64.len(),
                values.len()
            )));
        }

        let col_ptr: Vec<usize> = indptr_i64.iter().map(|&x| x as usize).collect();
        let row_idx: Vec<usize> = indices_i64.iter().map(|&x| x as usize).collect();
        let values_vec: Vec<f64> = values.to_vec();

        let sym = SymbolicSparseColMatRef::<usize>::new_checked(n, n, &col_ptr, None, &row_idx);
        let a = SparseColMatRef::<usize, f64>::new(sym, &values_vec);

        let params = CholeskySymbolicParams {
            supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SUPERNODAL,
            ..Default::default()
        };
        let symbolic = factorize_symbolic_cholesky::<usize>(
            sym,
            Side::Upper,
            SymmetricOrdering::Amd,
            params,
        )
        .map_err(|e| err(format!("symbolic factorization failed: {e:?}")))?;

        let mut l_values = vec![0.0_f64; symbolic.len_val()];
        let mut subdiag = vec![0.0_f64; n];
        let mut perm_fwd = vec![0_usize; n];
        let mut perm_inv = vec![0_usize; n];

        let lbl_params: Spec<LbltParams, f64> = Spec::default();
        let req = symbolic.factorize_numeric_intranode_lblt_scratch::<f64>(Par::Seq, lbl_params);
        let mut mem = MemBuffer::try_new(req).map_err(|_| err("OOM allocating scratch"))?;
        let stack = MemStack::new(&mut mem);

        let _ = symbolic.factorize_numeric_intranode_lblt::<f64>(
            &mut l_values,
            &mut subdiag,
            &mut perm_fwd,
            &mut perm_inv,
            a,
            Side::Upper,
            Par::Seq,
            stack,
            lbl_params,
        );

        Ok(SparseLDL {
            n,
            symbolic,
            l_values,
            subdiag,
            perm_fwd,
            perm_inv,
        })
    }

    fn solve<'py>(
        &self,
        py: Python<'py>,
        b: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let b_slice = b.as_slice().map_err(|e| err(format!("rhs: {e}")))?;
        if b_slice.len() != self.n {
            return Err(err(format!(
                "rhs length {} != n = {}",
                b_slice.len(),
                self.n
            )));
        }

        let mut x: Vec<f64> = b_slice.to_vec();
        let mat: MatMut<'_, f64> = MatMut::from_column_major_slice_mut(&mut x, self.n, 1);

        let perm_ref = unsafe {
            PermRef::<'_, usize>::new_unchecked(&self.perm_fwd, &self.perm_inv, self.n)
        };
        let ldl = IntranodeLbltRef::<usize, f64>::new(
            &self.symbolic,
            &self.l_values,
            &self.subdiag,
            perm_ref,
        );

        let req = self.symbolic.solve_in_place_scratch::<f64>(1, Par::Seq);
        let mut mem = MemBuffer::try_new(req).map_err(|_| err("OOM allocating solve scratch"))?;
        let stack = MemStack::new(&mut mem);

        ldl.solve_in_place_with_conj(Conj::No, mat, Par::Seq, stack);

        Ok(x.into_pyarray(py))
    }

    #[getter]
    fn n(&self) -> usize {
        self.n
    }

    #[getter]
    fn nnz_l(&self) -> usize {
        self.symbolic.len_val()
    }
}

#[pymodule]
fn faer_ldl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SparseLDL>()?;
    Ok(())
}
