use criterion::{black_box, criterion_group, criterion_main, Criterion};
use numpy::ndarray;
use rust_backend::linop::{CvxpyShape, Linop, LinopKind};
use rust_backend::{build_matrix, IdxMap};

fn generate_problem<'a>(m: i64) -> (IdxMap, IdxMap, IdxMap, i64, i64) {
    let n = m / 10;

    // Constraints for:
    // x = cp.Variable((n, 1))
    // objective = cp.Minimize(cp.sum_squares(cp.reshape(A @ x, (m, 1)) - b))
    // constraints = [0 <= x, x <= 1]

    let id_to_col = [(1i64, 0i64), (2i64, n), (-1i64, m + n)].into();
    let param_to_size = [(-1i64, 1i64)].into();
    let param_to_col = [(-1i64, 0i64)].into();
    let param_size_plus_one = 1;
    let var_length = m + n;
    (
        id_to_col,
        param_to_size,
        param_to_col,
        param_size_plus_one,
        var_length,
    )
}

fn get_linops(m: i64) -> Vec<Linop<'static>> {
    let n = m / 10;

    // create b = np.arange(m) and A = np.arange(m * n).reshape(m, n)

    let b = ndarray::Array2::from_shape_fn((m as usize, 1), |(i, _)| i as f64);
    let A = ndarray::Array2::from_shape_fn((m as usize, n as usize), |(i, j)| {
        (i * n as usize + j) as f64
    });

    let linops = vec![
        Linop {
            shape: CvxpyShape::D2(m as u64, 1),
            kind: LinopKind::Sum(vec![
                Linop {
                    shape: CvxpyShape::D2(m as u64, 1),
                    kind: LinopKind::Mul {
                        lhs: Box::new(Linop {
                            shape: CvxpyShape::D2(m as u64, n as u64),
                            kind: LinopKind::DenseConst(A),
                        }),
                        rhs: Box::new(Linop {
                            shape: CvxpyShape::D2(n as u64, 1),
                            kind: LinopKind::Variable(2),
                        }),
                    },
                },
                Linop {
                    shape: CvxpyShape::D2(m as u64, 1),
                    kind: LinopKind::DenseConst(b),
                },
                Linop {
                    shape: CvxpyShape::D2(m as u64, 1),
                    kind: LinopKind::Neg(Box::new(Linop {
                        shape: CvxpyShape::D2(m as u64, 1),
                        kind: LinopKind::Variable(1),
                    })),
                },
            ]),
        },
        Linop {
            shape: CvxpyShape::D2(n as u64, 1),
            kind: LinopKind::Sum(vec![
                Linop {
                    shape: CvxpyShape::D2(n as u64, 1),
                    kind: LinopKind::DenseConst(ndarray::Array2::zeros((n as usize, 1))),
                },
                Linop {
                    shape: CvxpyShape::D2(n as u64, 1),
                    kind: LinopKind::Variable(2),
                },
            ]),
        },
        Linop {
            shape: CvxpyShape::D2(n as u64, 1),
            kind: LinopKind::Sum(vec![
                Linop {
                    shape: CvxpyShape::D2(n as u64, 1),
                    kind: LinopKind::DenseConst(ndarray::Array2::ones((n as usize, 1))),
                },
                Linop {
                    shape: CvxpyShape::D2(n as u64, 1),
                    kind: LinopKind::Neg(Box::new(Linop {
                        shape: CvxpyShape::D2(n as u64, 1),
                        kind: LinopKind::Variable(2),
                    })),
                },
            ]),
        },
    ];
    linops
}

fn criterion_benchmark(c: &mut Criterion) {
    fn run_benchmark(m: i64, c: &mut Criterion) {
        let (id_to_col, param_to_size, param_to_col, param_size_plus_one, var_length) =
            generate_problem(m);
        c.bench_function(&m.to_string(), move |b| {
            b.iter(|| {
                build_matrix(
                    black_box(id_to_col.clone()),
                    black_box(param_to_size.clone()),
                    black_box(param_to_col.clone()),
                    black_box(param_size_plus_one),
                    black_box(var_length),
                    // Clone each Linop before collecting them into a new vector
                    black_box(get_linops(m)),
                )
            })
        });
    }

    run_benchmark(10, c);
    run_benchmark(100, c);
    run_benchmark(1000, c);
    // run_benchmark(10000, c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
