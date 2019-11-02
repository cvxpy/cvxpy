import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.variable import Variable


class SuppFuncAtom(Atom):

    def __init__(self, y, A, b, K_sels, x, cons):
        self.id = lu.get_id()
        eta = Variable(shape=(b.size,))
        self.args = [Atom.cast_to_const(y), Atom.cast_to_const(A), Atom.cast_to_const(b),
                     Atom.cast_to_const(eta)]
        horrible = self.args[2]
        horrible.__dict__['K_sels'] = K_sels
        horrible.__dict__['x'] = x
        horrible.__dict__['cons'] = cons
        self.validate_arguments()
        self._shape = tuple()
        pass

    def validate_arguments(self):
        if self.args[0].is_complex():
            raise ValueError("Arguments to SuppFuncAtom cannot be complex.")
        if not self.args[0].is_affine():
            raise ValueError("Arguments to SuppFuncAtom must be affine.")

    def variables(self):
        vars = self.args[0].variables() + self.args[3].variables()
        return vars

    def parameters(self):
        return []

    def constants(self):
        return []

    def shape_from_args(self):
        return self._shape

    def sign_from_args(self):
        return (False, False)

    def is_nonneg(self):
        return False

    def is_nonpos(self):
        return False

    def is_atom_convex(self):
        return True

    def is_atom_concave(self):
        return False

    def is_atom_affine(self):
        return False

    def is_atom_log_log_convex(self):
        return False

    def is_atom_log_log_concave(self):
        return False

    def is_atom_quasiconvex(self):
        return True

    def is_atom_quasiconcave(self):
        return False

    def _value_impl(self):
        from cvxpy.problems.problem import Problem
        from cvxpy.problems.objective import Maximize
        y_val = self.args[0].value.round(decimals=10).ravel(order='F')
        x_flat = self.args[2].x.flatten()
        cons = self.args[2].cons
        prob = Problem(Maximize(y_val @ x_flat), cons)
        val = prob.solve()
        return val
