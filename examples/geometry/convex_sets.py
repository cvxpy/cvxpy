# An object oriented approach to geometric problems with convex sets.
# Convex sets can be used as Variables.

import cvxpy as cvx
import cvxpy.lin_ops.lin_utils as lu
import numpy.linalg as la

class ConvexSet(cvx.Variable):
    # elem - a Variable representing an element of the set.
    # constr_func - a function that takes an affine objective and
    #               returns a list of affine constraints.
    def __init__(self, rows, cols, constr_func):
        self.constr_func = constr_func
        super(ConvexSet, self).__init__(rows, cols)

    # Applies the objective to the constr_func to get the affine constraints.
    def canonicalize(self):
        obj = lu.create_var(self.size, self.id)
        return (obj, self.constr_func(obj))

# Returns whether the value is contained in the set.
def contains(cvx_set, value):
    p = cvx.Problem(cvx.Minimize(0), [cvx_set == value])
    p.solve(solver=cvx.CVXOPT)
    return p.status == cvx.OPTIMAL

# Returns whether the set is empty.
def is_empty(cvx_set):
    return not contains(cvx_set, cvx_set)

# Returns the Euclidean distance between two sets.
def dist(lh_set, rh_set):
    objective = cvx.Minimize(cvx.norm(lh_set - rh_set, 2))
    return cvx.Problem(objective).solve(solver=cvx.CVXOPT)

# Returns the Euclidean projection of the value onto the set.
def proj(cvx_set, value):
    objective = cvx.Minimize(cvx.norm(cvx_set - value, 2))
    cvx.Problem(objective).solve(solver=cvx.CVXOPT)
    return cvx_set.value

# Returns a separating hyperplane between two sets
# in the form (normal, offset) where normal.T*x == offset
# for all x on the hyperplane.
def sep_hyp(lh_set, rh_set):
    w = cvx.Variable(*lh_set.size)
    p = cvx.Problem(cvx.Minimize(cvx.norm(w, 2)), [lh_set - rh_set == w])
    p.solve(solver=cvx.CVXOPT)
    # Normal vector to the hyperplane.
    normal = p.constraints[0].dual_value
    # A point on the hyperplane.
    point = (lh_set.value + rh_set.value)/2
    # The offset of the hyperplane.
    offset = normal.T*point
    return (normal, offset[0])

# Returns the intersection of two sets.
def intersect(lh_set, rh_set):
    def constr_func(aff_obj):
        # Combine the constraints from both sides and add an equality constraint.
        lh_obj, lh_constr = lh_set.canonical_form
        rh_obj, rh_constr = rh_set.canonical_form
        constraints = [lu.create_eq(aff_obj, lh_obj),
                       lu.create_eq(aff_obj, rh_obj),
        ]
        return constraints + lh_constr + rh_constr
    return ConvexSet(lh_set.size[0], lh_set.size[1], constr_func)

class Polyhedron(ConvexSet):
    # The set defined by Ax == b, Gx <= h.
    # G,h,A,b are numpy matrices or ndarrays.
    # The arguments A and b are optional.
    def __init__(self, G, h, A=None, b=None):
        G, h = map(self.cast_to_const, [G, h])
        def constr_func(aff_obj):
            G_aff = G.canonical_form[0]
            h_aff = h.canonical_form[0]
            Gx = lu.mul_expr(G_aff, aff_obj, h_aff.size)
            constraints = [lu.create_leq(Gx, h_aff)]
            if A is not None:
                A_const, b_const = map(self.cast_to_const, [A, b])
                A_aff = A_const.canonical_form[0]
                b_aff = b_const.canonical_form[0]
                Ax = lu.mul_expr(A_aff, aff_obj, b_aff.size)
                constraints += [lu.create_eq(Ax, b_aff)]
            return constraints
        super(Polyhedron, self).__init__(G.size[1], 1, constr_func)

class ConvexHull(ConvexSet):
    # The convex hull of a list of values.
    def __init__(self, values):
        values = map(self.cast_to_const, values)
        rows, cols = values[0].size
        def constr_func(aff_obj):
            theta = [lu.create_var((1, 1)) for i in xrange(len(values))]
            convex_objs = []
            for val, theta_var in zip(values, theta):
                val_aff = val.canonical_form[0]
                convex_objs.append(
                    lu.mul_expr(val_aff,
                                theta_var,
                                val_aff.size)
                )
            convex_combo = lu.sum_expr(convex_objs)
            one = lu.create_const(1, (1, 1))
            constraints = [lu.create_eq(aff_obj, convex_combo),
                           lu.create_eq(lu.sum_expr(theta), one)]
            for theta_var in theta:
                constraints.append(lu.create_geq(theta_var))
            return constraints
        super(ConvexHull, self).__init__(rows,cols,constr_func)
