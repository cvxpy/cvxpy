#![allow(non_snake_case)] // A lot of linear algebra in this file where we want capital matrices

use faer::{
    sparse::{SparseColMat, SparseColMatRef, SymbolicSparseColMat},
    ComplexField, Conjugate, Index, SimpleEntity,
};

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

pub fn to_triplets_iter<'a, I, E>(A: &'a SparseColMat<I, E>) -> impl Iterator<Item = (I, I, E)> + 'a
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

pub fn select_rows<'a, I, E>(A: SparseColMat<I, E>, rows: &[u64]) -> SparseColMat<I, E>
where
    I: Index + Copy + std::convert::From<u64> + std::convert::TryInto<usize>,
    E: SimpleEntity + Copy + Conjugate + ComplexField,
{
    let csr = A.to_row_major().unwrap();
    let mut triplets = Vec::new();

    for &i in rows {
        let row_index: I = i.try_into().unwrap(); // Assuming `I` can be constructed from `u64`
        for (j, v) in csr
            .col_indices_of_row(row_index as usize)
            .zip(csr.values_of_row(row_index as usize))
        {
            triplets.push((row_index, j, *v)); // Ensure `j` and `v` types match `I` and `E`
        }
    }
    SparseColMat::try_new_from_triplets(rows.len(), A.ncols(), &triplets).unwrap()
}
