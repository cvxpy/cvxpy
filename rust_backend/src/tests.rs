use faer::scale;
use faer::sparse::SparseColMat;
use faer::zipped;
use numpy::ndarray::ArrayView2;

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
    let variable_linop = Linop {
        shape: CvxpyShape::D2(2, 2),
        kind: LinopKind::Variable(1),
    };
    let empty_view = View::new(&context);

    let neg_linop = Linop {
        shape: CvxpyShape::D2(2, 2),
        kind: LinopKind::Neg(Box::new(variable_linop)),
    };

    let negated_view = process_constraints(&neg_linop, empty_view);
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

    let transpose_linop = Linop {
        shape: CvxpyShape::D2(2, 2),
        kind: LinopKind::Transpose(Box::new(linop)),
    };

    let transposed_view = process_constraints(&transpose_linop, view);
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

    let promote_linop = Linop {
        shape: CvxpyShape::D1(3),
        kind: LinopKind::Promote(Box::new(variable_linop)),
    };

    let promoted_view = process_constraints(&promote_linop, empty_view);
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
fn test_scalar_constant() {
    let linop = Linop {
        shape: CvxpyShape::D0,
        kind: LinopKind::ScalarConst(3.0),
    };
    let context = ViewContext {
        id_to_col: [(-1, 0)].into(),
        param_to_size: [(-1, 1)].into(),
        param_to_col: [(-1, 0)].into(),
        param_size_plus_one: 1,
        var_length: 0,
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

#[test]
fn test_dense_constant() {
    let mat = numpy::ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);

    let linop = Linop {
        shape: CvxpyShape::D2(2, 2),
        kind: LinopKind::DenseConst(mat),
    };

    let context = ViewContext {
        id_to_col: [(-1, 0)].into(),
        param_to_size: [(-1, 1)].into(),
        param_to_col: [(-1, 0)].into(),
        param_size_plus_one: 1,
        var_length: 0,
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
    let view_A = SparseColMat::try_new_from_triplets(4, 1, &triplets).unwrap();
    assert_eq!(
        view_A,
        SparseColMat::try_new_from_triplets(
            4,
            1,
            &[(0, 0, 1.0), (1, 0, 3.0), (2, 0, 2.0), (3, 0, 4.0)]
        )
        .unwrap()
    )
}

#[test]
fn test_sparse_constant() {
    let mat = faer::sparse::SparseColMat::try_new_from_triplets(
        2,
        2,
        &[(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0)],
    )
    .unwrap();

    let linop = Linop {
        shape: CvxpyShape::D2(2, 2),
        kind: LinopKind::SparseConst(&mat),
    };
    let context = ViewContext {
        id_to_col: [(-1, 0)].into(),
        param_to_size: [(-1, 1)].into(),
        param_to_col: [(-1, 0)].into(),
        param_size_plus_one: 1,
        var_length: 0,
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
    let view_A = SparseColMat::try_new_from_triplets(4, 1, &triplets).unwrap();
    assert_eq!(
        view_A,
        SparseColMat::try_new_from_triplets(4, 1, &[(0, 0, 1.0), (1, 0, 3.0), (2, 0, 2.0)])
            .unwrap()
    )
}

#[test]
fn test_mul() {
    let context = ViewContext {
        id_to_col: [(1, 0)].into(),
        param_to_size: [(-1, 1)].into(),
        param_to_col: [(-1, 0)].into(),
        param_size_plus_one: 1,
        var_length: 4,
    };
    let variable_linop = Linop {
        shape: CvxpyShape::D2(2, 2),
        kind: LinopKind::Variable(1),
    };
    let empty_view = View::new(&context);

    let mat = numpy::ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let lhs_linop = Linop {
        shape: CvxpyShape::D2(2, 2),
        kind: LinopKind::DenseConst(mat),
    };

    let mul_linop = Linop {
        shape: CvxpyShape::D2(2, 2),
        kind: LinopKind::Mul {
            lhs: Box::new(lhs_linop),
            rhs: Box::new(variable_linop),
        },
    };

    let out_view = process_constraints(&mul_linop, empty_view);

    let view_A = out_view.get_tensor_representation(0);
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

    // expected = np.array(
    //     [[1, 2, 0, 0],
    //      [3, 4, 0, 0],
    //      [0, 0, 1, 2],
    //      [0, 0, 3, 4]]
    // )

    let expected_A = SparseColMat::try_new_from_triplets(
        4,
        4,
        &[
            (0, 0, 1.0),
            (1, 0, 3.0),
            (0, 1, 2.0),
            (1, 1, 4.0),
            (2, 2, 1.0),
            (3, 2, 3.0),
            (2, 3, 2.0),
            (3, 3, 4.0),
        ],
    )
    .unwrap();
    assert_eq!(view_A, expected_A);
}
