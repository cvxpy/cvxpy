//! Minimal Python wrapper around faer's sparse symmetric Bunch-Kaufman
//! (intranode LBL^T) factorization.
//!
//! Surface:
//!   ldl = faer_ldl.SparseLDL(n, indptr, indices, data)
//!     -- input is the UPPER triangle of an n-by-n symmetric matrix
//!        in scipy CSC layout (col_ptr, row_idx, values).
//!   L_indptr, L_indices, L_data, D_diag, D_subdiag, perm = ldl.factor()
//!     -- L is unit lower-triangular, returned in CSC (with implicit
//!        unit diagonal). D is block-diagonal: D_diag[i] = D[i,i],
//!        D_subdiag[i] = D[i+1,i] (zero if 1x1 block at i).
//!     -- perm is a length-n forward permutation such that
//!        A[perm, :][:, perm] = (I + L) D (I + L)^T.
//!
//! We force the supernodal path (FORCE_SUPERNODAL) so the factorization
//! actually does Bunch-Kaufman pivoting. faer's simplicial path silently
//! degrades to plain LDLT (no pivoting), the same failure mode as qdldl.

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::lblt::factor::LbltParams;
use faer::perm::PermRef;
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, SymbolicCholesky,
    SymbolicCholeskyRaw, SymmetricOrdering,
};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Par, Side, Spec};
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

        if !matches!(symbolic.raw(), SymbolicCholeskyRaw::Supernodal(_)) {
            return Err(err(
                "internal: expected supernodal symbolic factor (FORCE_SUPERNODAL)",
            ));
        }

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

    /// Walk the supernodal storage and return CSC components of L (unit
    /// lower triangular, diagonal implicit), the diagonal and subdiagonal
    /// of D, and the composed permutation.
    fn factor<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<i64>>,
    )> {
        let n = self.n;
        let s_sym = match self.symbolic.raw() {
            SymbolicCholeskyRaw::Supernodal(s) => s,
            SymbolicCholeskyRaw::Simplicial(_) => {
                return Err(err("internal: simplicial path produced (expected supernodal)"));
            }
        };

        let n_super = s_sym.n_supernodes();
        let supernode_begin = s_sym.supernode_begin();
        let supernode_end = s_sym.supernode_end();
        let col_ptr_val = s_sym.col_ptr_for_val();
        let col_ptr_row = s_sym.col_ptr_for_row_idx();
        let row_idx_super = s_sym.row_idx();
        let nnz_per_super = s_sym.nnz_per_super();

        let mut d_diag = vec![0.0_f64; n];
        let d_subdiag: Vec<f64> = self.subdiag.clone();

        // Per-column count (for CSC indptr) plus triplet emit.
        let mut col_counts = vec![0_i64; n];
        let mut triplets: Vec<(i64, i64, f64)> = Vec::new();

        for s in 0..n_super {
            let s_start = supernode_begin[s];
            let s_end = supernode_end[s];
            let s_ncols = s_end - s_start;
            let pat_off = col_ptr_row[s];
            let pat_len = nnz_per_super[s];
            let pattern = &row_idx_super[pat_off..pat_off + pat_len];
            let s_nrows = s_ncols + pat_len;
            let blk = &self.l_values[col_ptr_val[s]..col_ptr_val[s + 1]];

            for c in 0..s_ncols {
                let global_c = s_start + c;
                // Diagonal of D
                d_diag[global_c] = blk[c + c * s_nrows];
                // Strict lower triangle within the diagonal block
                for r in (c + 1)..s_ncols {
                    let v = blk[r + c * s_nrows];
                    if v != 0.0 {
                        triplets.push(((s_start + r) as i64, global_c as i64, v));
                        col_counts[global_c] += 1;
                    }
                }
                // Off-diagonal block (pattern rows)
                for (i, &p_row) in pattern.iter().enumerate() {
                    let v = blk[(s_ncols + i) + c * s_nrows];
                    if v != 0.0 {
                        triplets.push((p_row as i64, global_c as i64, v));
                        col_counts[global_c] += 1;
                    }
                }
            }
        }

        // Compose permutations: perm_combined[i] = perm_amd[perm_intranode[i]]
        let perm_intra_fwd: &[usize] = &self.perm_fwd;
        let perm_combined: Vec<i64> = match self.symbolic.perm() {
            Some(p) => {
                let amd_fwd = p.arrays().0;
                (0..n)
                    .map(|i| amd_fwd[perm_intra_fwd[i]] as i64)
                    .collect()
            }
            None => perm_intra_fwd.iter().map(|&x| x as i64).collect(),
        };
        let _ = PermRef::<'_, usize>::new_checked; // keep import; nothing to do here.

        // Sort triplets by (col, row) and build CSC arrays.
        triplets.sort_unstable_by_key(|&(r, c, _)| (c, r));
        let nnz = triplets.len();
        let mut l_indptr = vec![0_i64; n + 1];
        for c in 0..n {
            l_indptr[c + 1] = l_indptr[c] + col_counts[c];
        }
        let mut l_indices = Vec::with_capacity(nnz);
        let mut l_data = Vec::with_capacity(nnz);
        for &(r, _c, v) in &triplets {
            l_indices.push(r);
            l_data.push(v);
        }

        Ok((
            l_indptr.into_pyarray(py),
            l_indices.into_pyarray(py),
            l_data.into_pyarray(py),
            d_diag.into_pyarray(py),
            d_subdiag.into_pyarray(py),
            perm_combined.into_pyarray(py),
        ))
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
