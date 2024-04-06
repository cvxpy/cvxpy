use faer::scale;
use faer::sparse::SparseColMat;
use faer::zipped;

use crate::backend::process_constraints;
use crate::backend::promote;
use crate::backend::transpose;
use crate::faer_ext;
use crate::linop::CvxpyShape;
use crate::linop::Linop;
use crate::linop::LinopKind;

use crate::view::View;
use crate::view::ViewContext;

#[test]
fn test_neg() {
    let context = ViewContext {
        id_to_col: [(1, 0), (2, 2)].into(),
        param_to_size: [(-1, 1), (3, 1)].into(),
        param_to_col: [(3, 0), (-1, 1)].into(),
        param_size_plus_one: 2,
        var_length: 4,
    };
    let linop = Linop {
        shape: CvxpyShape::D2(2, 2),
        kind: LinopKind::Variable(1),
    };
    let empty_view = View::new(&context);

    let view = process_constraints(&linop, empty_view);

    let view_A = view.get_tensor_representation(0);
    let mut triplets = Vec::new();
    for (r, c, d) in view_A
        .row
        .iter()
        .zip(&view_A.col)
        .zip(&view_A.data)
        .map(|((&r, &c), &d)| (r, c, d))
    {
        triplets.push((r, c, d));
    }

    let view_A = SparseColMat::try_new_from_triplets(4, 4, &triplets).unwrap();
    assert_eq!(view_A, faer_ext::eye(4));

    let negated_view = crate::backend::neg(&linop, view);
    let view_A = negated_view.get_tensor_representation(0);
    let mut triplets = Vec::new();
    for (r, c, d) in view_A
        .row
        .iter()
        .zip(&view_A.col)
        .zip(&view_A.data)
        .map(|((&r, &c), &d)| (r, c, d))
    {
        triplets.push((r, c, d));
    }

    let view_A = SparseColMat::try_new_from_triplets(4, 4, &triplets).unwrap();
    assert_eq!(view_A, -faer_ext::eye(4));
}

#[test]
fn test_transpose() {
    let context = ViewContext {
        id_to_col: [(1, 0), (2, 2)].into(),
        param_to_size: [(-1, 1), (3, 1)].into(),
        param_to_col: [(3, 0), (-1, 1)].into(),
        param_size_plus_one: 2,
        var_length: 4,
    };
    let linop = Linop {
        shape: CvxpyShape::D2(2, 2),
        kind: LinopKind::Variable(1),
    };
    let empty_view = View::new(&context);

    let view = process_constraints(&linop, empty_view);

    let transposed_view = transpose(&linop, view);
    let A = transposed_view.get_tensor_representation(0);
    let mut triplets = Vec::new();
    for (r, c, d) in A
        .row
        .iter()
        .zip(&A.col)
        .zip(&A.data)
        .map(|((&r, &c), &d)| (r, c, d))
    {
        triplets.push((r, c, d));
    }

    let A = SparseColMat::try_new_from_triplets(4, 4, &triplets).unwrap();
    let expected_A = SparseColMat::try_new_from_triplets(
        4,
        4,
        &[(0, 0, 1.0), (1, 2, 1.0), (2, 1, 1.0), (3, 3, 1.0)],
    )
    .unwrap();
    assert_eq!(A, expected_A);
}

#[test]
fn test_promote() {
    let context = ViewContext {
        id_to_col: [(1, 0), (2, 1)].into(),
        param_to_size: [(-1, 1), (3, 1)].into(),
        param_to_col: [(3, 0), (-1, 1)].into(),
        param_size_plus_one: 2,
        var_length: 4,
    };
    let variable_linop = Linop {
        shape: CvxpyShape::D1(1),
        kind: LinopKind::Variable(1),
    };
    let empty_view = View::new(&context);

    let view = process_constraints(&variable_linop, empty_view);

    let promote_linop = Linop {
        shape: CvxpyShape::D1(3),
        kind: LinopKind::Promote,
    };

    let promoted_view = promote(&promote_linop, view);
    let A = promoted_view.get_tensor_representation(0);
    let mut triplets = Vec::new();
    for (r, c, d) in A
        .row
        .iter()
        .zip(&A.col)
        .zip(&A.data)
        .map(|((&r, &c), &d)| (r, c, d))
    {
        triplets.push((r, c, d));
    }

    let A = SparseColMat::try_new_from_triplets(3, 1, &triplets).unwrap();
    let expected_A =
        SparseColMat::try_new_from_triplets(3, 1, &[(0, 0, 1.0), (1, 0, 1.0), (2, 0, 1.0)])
            .unwrap();
    assert_eq!(A, expected_A);
}

#[test]
fn test_constants() {
    // Test ScalarConst
    let linop = Linop {
        shape: CvxpyShape::D0,
        kind: LinopKind::ScalarConst(3.0),
    };
    let context = ViewContext {
        id_to_col: [(1, 0), (2, 2)].into(),
        param_to_size: [(-1, 1), (3, 1)].into(),
        param_to_col: [(3, 0), (-1, 1)].into(),
        param_size_plus_one: 2,
        var_length: 4,
    };
    let empty_view = View::new(&context);
    let view = process_constraints(&linop, empty_view);
    let view_A = view.get_tensor_representation(0);
    let mut triplets = Vec::new();
    for (r, c, d) in view_A
        .row
        .iter()
        .zip(&view_A.col)
        .zip(&view_A.data)
        .map(|((&r, &c), &d)| (r, c, d))
    {
        triplets.push((r, c, d));
    }
    let view_A = SparseColMat::try_new_from_triplets(1, 1, &triplets).unwrap();
    assert_eq!(view_A, scale(3.0) * &faer_ext::eye(1))
}