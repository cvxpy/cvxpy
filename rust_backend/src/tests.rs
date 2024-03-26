use faer::sparse::SparseColMat;
use faer::zipped;

use crate::faer_ext;
use crate::view::View;
use crate::view::ViewContext;
use crate::linop::CvxpyShape;
use crate::linop::LinopKind;
use crate::linop::Linop;
use crate::process_constraints;

#[test]
fn neg() {
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
    for (r, c, d) in view_A.row.iter().zip(&view_A.col).zip(&view_A.data).map(|((&r, &c), &d)| (r, c, d) ) {
        triplets.push((r, c, d));  
    }

    let view_A = SparseColMat::try_new_from_triplets(
        4, 4, &triplets
    ).unwrap();
    assert_eq!(view_A, faer_ext::eye(4));

    // view_A = view.get_tensor_representation(0)
    // view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
    // assert np.all(view_A == np.eye(4))

}
