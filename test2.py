import pickle
import cvxpy
with open("/Users/stevend2/Downloads/qp.pkl", "rb") as f:
        P, q, G, h, A, b = pickle.load(f)
size = 1229600
x = cvxpy.Variable(size)
prob = cvxpy.Problem(cvxpy.Minimize((1 / 2) * cvxpy.quad_form(x, P) + q.T @ x),
        [G @ x <= h,
        A @ x == b])
prob.solve(verbose=True)
