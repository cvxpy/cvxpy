# An object oriented approach to geometric problems with convex sets.

import cvxpy as cp

class ConvexSet(object):
    # elem - a Variable representing an element of the set.
    # constraints - constraints on elem.
    def __init__(self, elem, constraints):
        self.elem = elem
        self.constraints = constraints

    # Returns whether the value is contained in the set.
    def contains(self, value):
        p = cp.Problem(cp.Minimize(0), [self.elem == value] + self.constraints)
        return cp.Problem.get_status(p.solve()) == cp.SOLVED

    # Returns whether the set is empty.
    def is_empty(self):
        p = cp.Problem(cp.Minimize(0), self.constraints)
        return not cp.Problem.get_status(p.solve()) == cp.SOLVED

    # Returns the Euclidean projection of the value onto the set.
    def proj(self, value):
        objective = cp.Minimize(cp.norm2(self.elem - value))
        cp.Problem(objective, self.constraints).solve()
        return self.elem.value

    # Returns the Euclidean distance between two sets.
    # other - another ConvexSet.
    def dist(self, other):
        objective = cp.Minimize(cp.norm2(self.elem - other.elem))
        return cp.Problem(objective, self.constraints + other.constraints).solve()

    # Returns the intersection of two sets.
    # other - another ConvexSet.
    def intersect(self, other):
        constraints = self.constraints + other.constraints
        constraints += [self.elem == other.elem]
        return ConvexSet(self.elem, constraints)

class Polyhedron(ConvexSet):
    # The set defined by Ax == b, Gx <= h.
    # G,h,A,b are numpy matrices or ndarrays.
    # The arguments A and b are optional.
    def __init__(self, G, h, A=None, b=None):
        x = cp.Variable(G.shape[1])
        constraints = [G*x <= h]
        if A is not None:
            constraints += [A*x == b]
        super(Polyhedron, self).__init__(x, constraints)

class ConvexHull(ConvexSet):
    # The convex hull of a list of values.
    def __init__(self, values):
        theta = cp.Variable(len(values))
        convex_combo = sum(t*v for (t,v) in zip(theta, values))
        x = cp.Variable(*values[0].size)
        constraints = [x == convex_combo, sum(theta) == 1, theta >= 0]
        super(ConvexHull, self).__init__(x, constraints)