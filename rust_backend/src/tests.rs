use crate::view::View;
use crate::view::ViewContext;
use crate::linop::CvxpyShape;
use crate::linop::LinopKind;
use crate::linop::Linop;

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


}
