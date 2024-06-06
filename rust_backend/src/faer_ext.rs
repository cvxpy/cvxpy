#![allow(non_snake_case)] // A lot of linear algebra in this file where we want capital matrices

use faer::{
    sparse::{SparseColMat, SparseColMatRef, SymbolicSparseColMat},
    ComplexField, Conjugate, Index, SimpleEntity,
};

use crate::SparseMatrix;

pub fn dense_to_sparse(A: &numpy::ndarray::Array2<f64>) -> SparseMatrix {
    let numel_usize = A.nrows() * A.ncols();
    let numel = u64::try_from(numel_usize).unwrap();
    let col_ptr = vec![0, numel];
    let row_idx = (0..numel).collect();

    let row_val = A.columns().into_iter().flat_map(|column| column.into_iter().copied()).collect();
    // Safety: by construction
    let symbolic = unsafe {
        SymbolicSparseColMat::new_unchecked(numel_usize, 1, col_ptr, Some(vec![numel]), row_idx)
    };

    SparseColMat::new(symbolic, row_val)
}

/*
pub fn reshape<I: Index, E: SimpleEntity>(A: SparseColMatRef<'_, I, E>,
                                               (m, n): (I, I)) -> SparseColMat<I, E> {
    //! Reshape A into (m,n) in Fortran (column-major) order.
    let oldn: I = A.ncols();
    let mut triplets: Vec<(I, I, E)> = Vec::with_capacity(A.compute_nnz()); // Check this is the
                                                                            // write method
    for oldi in 0..oldn {
        for (oldj, v) in A.col_indices_of_row(oldi).zip(A.values_of_row(oldi)) {
            triplets.push((oldj * oldn + oldi) % m, (oldj * oldn + oldi) / m, *v);
        }
    }
    SparseColMat::try_new_from_triplets(m, n, triplets).unwrap()
} */

pub fn eye(n: u64) -> SparseColMat<u64, f64> {
    let n_usize = n.try_into().unwrap();
    SparseColMat::new(
        SymbolicSparseColMat::<u64>::new_checked(
            n_usize,
            n_usize,
            (0..n + 1).collect::<Vec<_>>(),
            Some([1u64].repeat(n_usize)),
            (0..n).collect::<Vec<_>>(),
        ),
        [1.0f64].repeat(n_usize),
    )
}

pub fn to_triplets_iter<I, E>(A: &SparseColMat<I, E>) -> impl Iterator<Item = (I, I, E)> + '_
where
    I: Index + TryFrom<usize> + Copy,
    E: SimpleEntity + Copy,
    <I as TryFrom<usize>>::Error: std::fmt::Debug,
{
    (0..A.ncols()).flat_map(move |j| {
        let col_index = j.try_into().unwrap();
        A.row_indices_of_col(j)
            .zip(A.values_of_col(j))
            .map(move |(i, &v)| (i.try_into().unwrap(), col_index, v))
    })
}

pub fn select_rows(A: &SparseColMat<u64, f64>, rows: &[u64]) -> SparseColMat<u64, f64> {
    let csr = A.to_row_major().unwrap();
    let mut triplets = Vec::new();

    for (i, &r) in rows.iter().enumerate() {
        for (j, &v) in csr
            .col_indices_of_row(r as usize)
            .zip(csr.values_of_row(r as usize))
        {
            triplets.push((i as u64, j as u64, v));
        }
    }
    SparseColMat::try_new_from_triplets(rows.len(), A.ncols(), &triplets).unwrap()
}

/* pub(crate) fn identity_kron(reps: u64, lhs: &SparseColMat<u64, f64>) -> SparseColMat<u64, f64> {
    if reps == 1 {
        lhs.clone()
    } else {
        let mut triplets = Vec::with_capacity(lhs.compute_nnz() * reps as usize);
        for rep in 0..reps {
            for (r, c, d) in to_triplets_iter(lhs) {
                triplets.push((
                    r + rep * lhs.nrows() as u64,
                    c + rep * lhs.ncols() as u64,
                    d,
                ));
            }
        }
        SparseColMat::try_new_from_triplets(
            reps as usize * lhs.nrows(),
            reps as usize * lhs.ncols(),
            &triplets,
        )
        .unwrap()
    }
} */

pub(crate) fn identity_kron(reps: u64, A: &SparseMatrix) -> SparseColMat<u64, f64> {
    if reps == 1 {
        A.clone()
    } else {
        let capacity = A.compute_nnz() * reps as usize;
        let mut row_indices = Vec::with_capacity(capacity + 1);
        let mut entries = Vec::with_capacity(capacity + 1);
        let mut col_ptrs = Vec::with_capacity(reps as usize * A.ncols() + 2);
        let mut entries_so_far = 0;
        col_ptrs.push(entries_so_far);
        for r in 0..reps {
            for j in 0..A.ncols() {
                for (i, value) in A.row_indices_of_col(j).zip(A.values_of_col(j)) {
                    row_indices.push(i as u64 + r * A.nrows() as u64);
                    entries.push(*value);
                    entries_so_far += 1;
                }
                col_ptrs.push(entries_so_far);
            }
        }
        SparseColMat::new(
            // SAFETY: By construction
            unsafe {
                SymbolicSparseColMat::new_unchecked(
                    reps as usize * A.nrows(),
                    reps as usize * A.ncols(),
                    col_ptrs,
                    None,
                    row_indices,
                )
            },
            entries,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_kron_p1() {
        let mat =
            SparseMatrix::try_new_from_triplets(2, 2, &[(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0)])
                .unwrap();

        let result = identity_kron(1, &mat);

        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 2);
        assert_eq!(result.compute_nnz(), 3);
        assert_eq!(
            to_triplets_iter(&result).collect::<Vec<_>>(),
            vec![(0, 0, 1.0), (1, 0, 3.0), (0, 1, 2.0)]
        );
    }

    #[test]
    fn test_identity_kron_p3() {
        let mat =
            SparseMatrix::try_new_from_triplets(2, 2, &[(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0)])
                .unwrap();

        let result = identity_kron(3, &mat);

        assert_eq!(result.nrows(), 6);
        assert_eq!(result.ncols(), 6);
        assert_eq!(result.compute_nnz(), 9);
        assert_eq!(
            to_triplets_iter(&result).collect::<Vec<_>>(),
            vec![
                (0, 0, 1.0),
                (1, 0, 3.0),
                (0, 1, 2.0),
                (2, 2, 1.0),
                (3, 2, 3.0),
                (2, 3, 2.0),
                (4, 4, 1.0),
                (5, 4, 3.0),
                (4, 5, 2.0)
            ]
        );
    }
}
