from noncvx_variable import NonCvxVariable
import cvxpy
import cvxopt

# Use ADMM to attempt non-convex problem.
def admm(self, rho=0.5, max_iter=5):
    objective,eq_constr,ineq_constr,dims = self.canonicalize()
    variables = self.variables(objective, eq_constr + ineq_constr)
    noncvx_vars = []
    for obj in variables:
        if isinstance(obj,NonCvxVariable):
            # Initialize replicant z and residual u.
            z = cvxpy.Parameter(*obj.size)
            z.value = cvxopt.matrix(0, obj.size, tc='d')
            u = cvxpy.Parameter(*obj.size)
            u.value = cvxopt.matrix(0, obj.size, tc='d')
            noncvx_vars += [(obj, z, u)]
    # Form ADMM problem.
    obj = self.objective.expr
    for x,z,u in noncvx_vars:
        obj = obj + (rho/2)*sum(cvxpy.square(x - z + u))
    p = cvxpy.Problem(cvxpy.Minimize(obj), self.constraints)
    # ADMM loop
    for i in range(max_iter):
        p.solve()
        for x,z,u in noncvx_vars:
            z.value = x.round(x.value + u.value)
            u.value = x.value - z.value
    # Fix noncvx variables and solve.
    fix_constr = []
    for x,z,u in noncvx_vars:
        print z.value
        fix_constr += x.fix(z.value)
    p = cvxpy.Problem(self.objective, self.constraints + fix_constr)
    return p.solve()

# Add admm method to cvxpy Problem.
cvxpy.Problem.register_solve("admm", admm)