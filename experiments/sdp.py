import numpy as np
import cvxpy as cp

def scs_data_from_cvxpy_problem(problem):
    data = problem.get_problem_data(cp.SCS)[0]
    cone_dims = cp.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(data[
                                                                                  "dims"])
    return data["A"], data["b"], data["c"], cone_dims


def randn_symm(n):
    A = np.random.randn(n, n)
    return (A + A.T) / 2


def randn_psd(n):
    A = 1. / 10 * np.random.randn(n, n)
    return A@A.T


n = 200
p = 300
# Generate problem data
C = randn_psd(n)
As = [randn_symm(n) for _ in range(p)]
Bs = np.random.randn(p)

# Extract problem data using cvxpy
import cProfile
pr = cProfile.Profile()
pr.enable()
X = cp.Variable((n, n), PSD=True)
objective = cp.trace(C@X)
constraints = [cp.trace(As[i]@X) == Bs[i] for i in range(p)]
prob = cp.Problem(cp.Minimize(objective), constraints)
A, b, c, cone_dims = scs_data_from_cvxpy_problem(prob)
pr.disable()
pr.dump_stats("sdp_canon_200_300.pf")
