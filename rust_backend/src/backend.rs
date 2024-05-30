use std::collections::HashMap;
use std::hash::Hash;

use faer::sparse::SparseColMat;

use crate::faer_ext;
use crate::linop::CvxpyShape;
use crate::linop::Linop;
use crate::linop::LinopKind;
use crate::view::Tensor;
use crate::view::View;
use crate::SparseMatrix;

pub(crate) const CONST_ID: i64 = -1;

fn get_variable_tensor(shape: &CvxpyShape, id: i64) -> Tensor {
    assert!(id > CONST_ID);
    let n = shape.numel();
    [(id, [(CONST_ID, faer_ext::eye(n))].into())].into()
}

pub(crate) fn process_constraints<'a>(linop: &Linop<'a>, view: View<'a>) -> View<'a> {
    match linop.kind {
        LinopKind::Variable(id) => View {
            variables: [id].into(),
            tensor: get_variable_tensor(&linop.shape, id),
            is_parameter_free: true,
            context: view.context,
        },
        LinopKind::ScalarConst(c) => {
            let mat = SparseColMat::try_new_from_triplets(1, 1, &[(0, 0, c)]).unwrap();
            let tensor = [(CONST_ID, [(CONST_ID, mat)].into())].into();
            View {
                variables: [CONST_ID].into(),
                tensor,
                is_parameter_free: true,
                context: view.context,
            }
        }
        LinopKind::DenseConst(ref mat) => {
            let mut triplets = Vec::with_capacity(mat.ncols() * mat.nrows());
            for ((i, j), v) in mat.indexed_iter() {
                triplets.push((i as u64 + j as u64 * mat.nrows() as u64, 0_u64, *v));
            }
            let mat = SparseColMat::try_new_from_triplets(mat.nrows() * mat.ncols(), 1, &triplets)
                .unwrap();
            let tensor = [(CONST_ID, [(CONST_ID, mat)].into())].into();
            View {
                variables: [CONST_ID].into(),
                tensor,
                is_parameter_free: true,
                context: view.context,
            }
        }
        LinopKind::SparseConst(mat) => {
            let mut triplets = Vec::with_capacity(mat.compute_nnz());
            for (i, j, v) in faer_ext::to_triplets_iter(mat) {
                triplets.push((i + j * mat.nrows() as u64, 0, v));
            }
            let mat = SparseColMat::try_new_from_triplets(mat.nrows() * mat.ncols(), 1, &triplets)
                .unwrap();
            let tensor = [(CONST_ID, [(CONST_ID, mat)].into())].into();
            View {
                variables: [CONST_ID].into(),
                tensor,
                is_parameter_free: true,
                context: view.context,
            }
        }
        // LinOpKind::Hstack => hstack(linop, view),
        // LinOpKind::Vstack => vstack(linop, view),
        _ => panic!(),
    }
}

pub(crate) fn parse_args<'a>(linop: &Linop<'a>, view: View<'a>) -> View<'a> {
    let mut res: Option<View> = None;
    for arg in linop.args.iter() {
        let arg_coeff = process_constraints(arg, view.clone());
        let arg_res = match arg.kind {
            LinopKind::Neg => neg(view),
            LinopKind::Transpose => transpose(linop, view),
            LinopKind::Sum => view, // Sum (along axis 1) is implicit in Ax+b, so it is a NOOP.
            LinopKind::Reshape => view, // Reshape is a NOOP.
            LinopKind::Promote => promote(linop, view),
            LinopKind::Mul { ref lhs } => mul(lhs, view),
            _ => panic!(),
        };
        res = if let Some(res) = res {
            res.add_inplace(&arg_res);
            Some(res)
        } else {
            Some(arg_res)
        }
    }
    res.unwrap()
}

pub(crate) fn mul<'a>(lhs: &Linop<'a>, view: View<'a>) -> View<'a> {
    let lhs = get_constant_data(lhs, &view, false);

    let is_parameter_free;
    let func;

    match lhs.keys().copied().collect::<Vec<_>>().as_slice() {
        [CONST_ID] => {
            let lhs = &lhs[&CONST_ID];
            is_parameter_free = true;
            let reps = view.rows() / lhs.ncols() as u64;
            let stacked_lhs = faer_ext::identity_kron(reps, lhs);
            func = move |x: &SparseMatrix, p: u64| -> SparseMatrix {
                faer::sparse::linalg::matmul::sparse_sparse_matmul(
                    faer_ext::identity_kron(p, &stacked_lhs).as_ref(),
                    x.as_ref(),
                    1.0,
                    faer::Parallelism::None,
                )
                .unwrap()
            };
        }
        _ => panic!(),
    }
    view.accumulate_over_variables(&func, is_parameter_free)
}

pub(crate) fn get_constant_data<'a>(
    linop: &Linop<'a>,
    view: &View<'a>,
    column: bool,
) -> HashMap<i64, SparseMatrix> {
    // TODO: Add fast path for when linop is a constant

    let mut constant_view = process_constraints(linop, view.clone());
    assert!(constant_view.variables == [CONST_ID].into());
    let constant_data = constant_view.tensor.remove(&CONST_ID).unwrap();

    if !column {
        match linop.shape {
            CvxpyShape::D0 => constant_data,
            CvxpyShape::D1(n) => reshape_constant_data(constant_data, (1, n)),
            CvxpyShape::D2(m, n) => reshape_constant_data(constant_data, (m, n)),
        }
    } else {
        constant_data
    }
}

pub(crate) fn reshape_constant_data(
    constant_data: HashMap<i64, SparseMatrix>,
    shape: (u64, u64),
) -> HashMap<i64, SparseMatrix> {
    constant_data
        .into_iter()
        .map(|(k, v)| (k, reshape_single_constant_tensor(v, shape)))
        .collect()
}

pub(crate) fn reshape_single_constant_tensor(v: SparseMatrix, (m, n): (u64, u64)) -> SparseMatrix {
    assert!(v.ncols() == 1);
    let p = v.nrows() as u64 / (m * n);
    let n_old_rows = v.nrows() as u64 / p;

    // TODO: exploit column format and the fact that it is a single column vec
    let triplets: Vec<(u64, u64, f64)> = faer_ext::to_triplets_iter(&v)
        .map(|(i, _, d)| {
            let row = i % n_old_rows;
            let slice = i / n_old_rows;

            let new_row = slice * m + row % m;
            let new_col = row / m;

            (new_row, new_col, d)
        })
        .collect();

    SparseColMat::try_new_from_triplets((p * m) as usize, n as usize, &triplets).unwrap()
}

pub(crate) fn neg(mut view: View<'_>) -> View<'_> {
    // Second argument is not used for neg
    view.apply_all(|x, _p| -x);
    view
}

pub(crate) fn transpose<'a>(linop: &Linop, mut view: View<'a>) -> View<'a> {
    let rows = get_transpose_rows(&linop.shape);
    view.select_rows(&rows);
    view
}

pub(crate) fn get_transpose_rows(shape: &CvxpyShape) -> Vec<u64> {
    let (m, n) = shape.broadcasted_shape();
    let rows: Vec<u64> = (0..n)
        .flat_map(|j| (0..m).map(move |i| i * n + j))
        .collect();
    rows
}

pub(crate) fn promote<'a>(linop: &Linop, mut view: View<'a>) -> View<'a> {
    let rows = vec![0; linop.shape.numel() as usize];
    view.select_rows(&rows);
    view
}

#[cfg(test)]
mod test_backend {
    use super::*;

    #[test]
    fn test_get_transpose_rows() {
        let shape = CvxpyShape::D2(3, 2);
        let rows = get_transpose_rows(&shape);
        assert_eq!(rows, vec![0, 2, 4, 1, 3, 5]);

        let shape = CvxpyShape::D2(2, 3);
        let rows = get_transpose_rows(&shape);
        assert_eq!(rows, vec![0, 3, 1, 4, 2, 5]);

        let shape = CvxpyShape::D1(3);
        let rows = get_transpose_rows(&shape);
        assert_eq!(rows, vec![0, 1, 2]);

        let shape = CvxpyShape::D0;
        let rows = get_transpose_rows(&shape);
        assert_eq!(rows, vec![0]);
    }
}
