from cvxpy.reductions import Reduction
from 

class Dcp2Cone(Reduction):


    def accepts(self, prob):
        return prob.is_dcp()



    def apply(self, prob):
        obj, constrs = canonicalize_tree(problem.):
        for c in prob.constraints:
             new_c = canonicalize_constr(c)
             constrs += c
        return 



    @static_method
    def canonicalize_expr(expr):
        canon_args = []
        for arg in expr.args:
            canon_args += [canonicalize_expr]
        canon_expr = CANON[type(expr)](expr)
        return canon_expr



    def store_vars(self):



    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """

        solution = 


        return None
