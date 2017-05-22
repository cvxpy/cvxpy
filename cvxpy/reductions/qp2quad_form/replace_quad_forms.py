from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.expressions.variables import Variable
from cvxpy.lin_ops.lin_op import LinOp, NO_OP
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.reductions.reduction import Reduction

class ReplaceQuadForms(Reduction):
    """Replaces symbolic quadratic forms in the objective by ordinary variables
    """

    def accepts(self, problem):
        return problem.objective.is_quadratic() or problem.objective.is_affine()

    def apply(self, problem):
        expr = problem.objective.expr
        # Insert no-op such that root is never a quadratic form
        root = LinOp(NO_OP, expr.shape, [expr], [])
        quad_forms = self.replace_quad_forms(root, {})
        
        # root is now an affine expression
        if isinstance(problem.objective, Minimize):
           new_obj = Minimize(root.args[0])
        elif isinstance(problem.objective, Maximize):
           new_obj = Maximize(root.args[0])
        
        new_problem = Problem(new_obj, [])
        return new_problem, quad_forms

    def invert(self, problem, quad_forms):
        for _, quad_form_info in quad_forms.items():
            expr, idx, quad_form = quad_form_info
            expr.args[idx] = quad_form
        return problem

    def replace_quad_forms(self, expr, quad_forms):
        for idx, arg in enumerate(expr.args):
            if isinstance(arg, SymbolicQuadForm):
                quad_forms = self.replace_quad_form(expr, idx, quad_forms)
            else:
                quad_forms = self.replace_quad_forms(arg, quad_forms)
        return quad_forms

    def replace_quad_form(self, expr, idx, quad_forms):
        quad_form = expr.args[idx]
        placeholder = Variable(*quad_form.shape)
        expr.args[idx] = placeholder
        quad_forms[placeholder.id] = (expr, idx, quad_form)
        return quad_forms