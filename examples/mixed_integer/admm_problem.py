from card_variable import CardVariable
import cvxpy

# Solve method with different options. 
def solve(self, method=None, **kwargs):
    if method is None:
        self._solve(**kwargs)
    elif method == "admm":
        self._admm(**kwargs)

# Use ADMM to attempt non-convex problem.
def admm(self, rho=0.5, max_iter=5):
    objective,eq_constr,ineq_constr,dims = self.canonicalize()
    variables = self.variables(objective, eq_constr + ineq_constr)
    noncvx_vars = [obj for obj in variables 
                   if isinstance(obj,CardVariable)]
    # Form ADMM problem.
    reg_obj = (rho/2)*sum(v.reg_obj() for v in noncvx_vars)
    p = cvxpy.Problem(cvxpy.Minimize(self.objective.expr + reg_obj), 
                      self.constraints)
    # ADMM loop
    for i in range(max_iter):
        p.solve()
        for var in noncvx_vars:
            var.project()
            var.update()
    # Fix noncvx variable and solve.
    fix_constr = []
    map(fix_constr.extend, (v.fix() for v in noncvx_vars))
    p = cvxpy.Problem(self.objective, self.constraints + fix_constr)
    return p.solve()

# Add admm method to cvxpy Problem.
cvxpy.Problem._solve = cvxpy.Problem.solve
cvxpy.Problem._admm = admm
cvxpy.Problem.solve = solve
