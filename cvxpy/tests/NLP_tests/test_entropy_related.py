import numpy as np
import numpy.linalg as LA

import cvxpy as cp


class TestStressMLE():

    # convex problem, standard entropy f(x) = - x log x.
    def test_entropy_one(self):
       np.random.seed(0)
       n = 100
       q = cp.Variable(n, nonneg=True)
       A = np.random.rand(n, n)
       obj = cp.sum(cp.entr(A @ q))
       constraints = [cp.sum(q) == 1]
       problem = cp.Problem(cp.Maximize(obj), constraints)
       problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
       q_opt_nlp = q.value 
       problem.solve(solver=cp.CLARABEL, verbose=True)
       q_opt_clarabel = q.value
       assert(LA.norm(q_opt_nlp - q_opt_clarabel) <= 1e-4)


    # nonconvex problem, compute minimum entropy distribution
    # over simplex (the analytical solution is any of the vertices)
    def test_entropy_two(self):
        np.random.seed(0)
        n = 10
        q = cp.Variable((n, ), nonneg=True)
        q.value = np.random.rand(n)
        q.value = q.value / np.sum(q.value)
        obj = cp.sum(cp.entr(q))
        constraints = [cp.sum(q) == 1]
        problem = cp.Problem(cp.Minimize(obj), constraints)
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True, 
                      hessian_approximation='limited-memory')
        q_opt_nlp = q.value 
        assert(np.sum(q_opt_nlp > 1e-8) == 1)

    # convex formulation, relative entropy f(x, y) = x log (x / y)
    def test_rel_entropy_one(self):
        np.random.seed(0)
        n = 40
        p = np.random.rand(n, )
        p = p / np.sum(p)
        q = cp.Variable(n, nonneg=True)
        A = np.random.rand(n, n)
        obj = cp.sum(cp.rel_entr(A @ q, p))
        constraints = [cp.sum(q) == 1]
        problem = cp.Problem(cp.Minimize(obj), constraints)
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        q_opt_nlp = q.value 
        problem.solve(solver=cp.CLARABEL, verbose=True)
        q_opt_clarabel = q.value
        assert(LA.norm(q_opt_nlp - q_opt_clarabel) <= 1e-4)

    def test_rel_entropy_one_switched_arguments(self):
        np.random.seed(0)
        n = 40
        p = np.random.rand(n, )
        p = p / np.sum(p)
        q = cp.Variable(n, nonneg=True)
        A = np.random.rand(n, n)
        obj = cp.sum(cp.rel_entr(p, A @ q))
        constraints = [cp.sum(q) == 1]
        problem = cp.Problem(cp.Minimize(obj), constraints)
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        q_opt_nlp = q.value 
        problem.solve(solver=cp.CLARABEL, verbose=True)
        q_opt_clarabel = q.value
        assert(LA.norm(q_opt_nlp - q_opt_clarabel) <= 1e-4)

    def test_KL_one(self):
        np.random.seed(0)
        n = 40
        p = np.random.rand(n, )
        p = p / np.sum(p)
        q = cp.Variable(n, nonneg=True)
        A = np.random.rand(n, n)
        obj = cp.sum(cp.kl_div(A @ q, p))
        constraints = [cp.sum(q) == 1]
        problem = cp.Problem(cp.Minimize(obj), constraints)
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        q_opt_nlp = q.value 
        problem.solve(solver=cp.CLARABEL, verbose=True)
        q_opt_clarabel = q.value
        assert(LA.norm(q_opt_nlp - q_opt_clarabel) <= 1e-4)

    def test_KL_two(self):
        np.random.seed(0)
        n = 40
        p = np.random.rand(n, )
        p = p / np.sum(p)
        q = cp.Variable(n, nonneg=True)
        A = np.random.rand(n, n)
        obj = cp.sum(cp.kl_div(p, A @ q))
        constraints = [cp.sum(q) == 1]
        problem = cp.Problem(cp.Minimize(obj), constraints)
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        q_opt_nlp = q.value 
        problem.solve(solver=cp.CLARABEL, verbose=True)
        q_opt_clarabel = q.value
        assert(LA.norm(q_opt_nlp - q_opt_clarabel) <= 1e-4)


    # nonnegative matrix factorization with KL objective (nonconvex)
    def test_KL_three(self):
        np.random.seed(0)
        n, m, k = 40, 20, 4
        X_true = np.random.rand(n, k)
        Y_true = np.random.rand(k, m)
        A = X_true @ Y_true 
        A = np.clip(A, 0, None)
        X = cp.Variable((n, k), bounds=[0, None])
        Y = cp.Variable((k, m), bounds=[0, None])
        # without random initialization we converge to a very structured
        # point that is not the global minimizer 
        X.value = np.random.rand(n, k)
        Y.value = np.random.rand(k, m)
        obj = cp.sum(cp.kl_div(A, X @ Y))
        problem = cp.Problem(cp.Minimize(obj))
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        assert(obj.value <= 1e-10)
