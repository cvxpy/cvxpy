# An object oriented approach to geometric problems with convex sets.
# Convex sets can be used as Variables.

from cvxpy.expressions.affine import AffObjective
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
import cvxpy as cp
import numpy.linalg as la

class ConvexSet(cp.Variable):
    # elem - a Variable representing an element of the set.
    # constr_func - a function that takes an affine objective and
    #               returns a list of affine constraints.
    def __init__(self, rows, cols, constr_func):
        self.constr_func = constr_func
        super(ConvexSet, self).__init__(rows, cols)

    # Applies the objective to the constr_func to get the affine constraints.
    def _constraints(self):
        return self.constr_func(self._objective())

# Returns whether the value is contained in the set.
def contains(cvx_set, value):
    p = cp.Problem(cp.Minimize(0), [cvx_set == value])
    return cp.get_status(p.solve()) == cp.SOLVED

# Returns whether the set is empty.
def is_empty(cvx_set):
    return not contains(cvx_set, cvx_set)

# Returns the Euclidean distance between two sets.
def dist(lh_set, rh_set):
    objective = cp.Minimize(cp.norm2(lh_set - rh_set))
    return cp.Problem(objective).solve()

# Returns the Euclidean projection of the value onto the set.
def proj(cvx_set, value):
    objective = cp.Minimize(cp.norm2(cvx_set - value))
    cp.Problem(objective).solve()
    return cvx_set.value

# Returns a separating hyperplane between two sets
# in the form (normal,offset) where normal.T*x == offset
# for all x on the hyperplane.
def sep_hyp(lh_set, rh_set):
    w = cp.Variable(*lh_set.size)
    p = cp.Problem(cp.Minimize(cp.norm2(w)), [lh_set - rh_set == w])
    p.solve()
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
        lh_obj,lh_constr = lh_set.canonical_form()
        rh_obj,rh_constr = rh_set.canonical_form()
        constraints = [AffEqConstraint(aff_obj, lh_obj),
                       AffEqConstraint(aff_obj, rh_obj),
        ]
        return constraints + lh_constr + rh_constr
    return ConvexSet(lh_set.size[0], lh_set.size[1], constr_func)

class Polyhedron(ConvexSet):
    # The set defined by Ax == b, Gx <= h.
    # G,h,A,b are numpy matrices or ndarrays.
    # The arguments A and b are optional.
    def __init__(self, G, h, A=None, b=None):
        G = self.cast_to_const(G)
        def constr_func(aff_obj):
            G_aff = G._objective()
            constraints = [AffLeqConstraint(G_aff*aff_obj, h)]
            if A is not None:
                A_aff = self.cast_to_const(A)._objective()
                constraints += [AffEqConstraint(A_aff*aff_obj, b)]
            return constraints
        super(Polyhedron, self).__init__(G.size[1], 1, constr_func)

class ConvexHull(ConvexSet):
    # The convex hull of a list of values.
    def __init__(self, values):
        values = map(self.cast_to_const, values)
        rows,cols = values[0].size
        def constr_func(aff_obj):
            theta = cp.Variable(len(values))
            convex_combo = sum(v*t for (t,v) in zip(theta, values))
            constraints = [AffEqConstraint(aff_obj, convex_combo),
                           AffEqConstraint(sum(theta), 1), 
                           AffLeqConstraint(0, theta)]
            return constraints
        super(ConvexHull, self).__init__(rows,cols,constr_func)