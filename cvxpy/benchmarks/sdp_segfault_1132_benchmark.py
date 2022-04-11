import numpy as np

import cvxpy as cp
from cvxpy.benchmarks.benchmark import Benchmark


class SDPSegfault1132Benchmark(Benchmark):

    @staticmethod
    def name() -> str:
        return "SDP Segfault issue 1132"

    @staticmethod
    def data_available(download_missing_data: bool) -> bool:
        return True

    @staticmethod
    def get_problem_instance() -> cp.Problem:
        n = 100
        alpha = 1
        np.random.seed(0)
        points = np.random.rand(5, n)
        xtx = points.T @ points
        xtxd = np.diag(xtx)
        e = np.ones((n,))
        D = np.outer(e, xtxd) - 2 * xtx + np.outer(xtxd, e)
        # Construct W
        W = np.ones((n, n))

        # Define V and e
        n = D.shape[0]
        x = -1 / (n + np.sqrt(n))
        y = -1 / np.sqrt(n)
        V = np.ones((n, n - 1))
        V[0, :] *= y
        V[1:, :] *= x
        V[1:, :] += np.eye(n - 1)
        e = np.ones((n, 1))

        # Solve optimization problem
        G = cp.Variable((n - 1, n - 1), PSD=True)
        objective = cp.Maximize(cp.trace(G) - alpha *
                                cp.norm(cp.multiply(W, cp.kron(e, cp.reshape(cp.diag(V @ G @ V.T),

                                                                             (1, n))) + cp.kron(
                                    e.T,
                                    cp.reshape(
                                        cp.diag(
                                            V @ G @ V.T),
                                        (
                                            n,
                                            1))) - 2 * V @ G @ V.T - D),
                                        p='fro'))
        problem = cp.Problem(objective, [])
        return problem


if __name__ == '__main__':
    bench = SDPSegfault1132Benchmark()
    bench.run_benchmark()
    bench.print_benchmark_results()
