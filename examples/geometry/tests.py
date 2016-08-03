# Problems involving polyhedra.

import convex_sets as cs
import numpy as np

n = 2
m = 2*n
A1 = np.matrix("1 1; 1 -1; -1 1; -1 -1")
A2 = np.matrix("1 0; -1 0; 0 1; 0 -1")
b1 = 2*np.ones((m,1))
b2 = np.matrix("5; -3; 4; -2")

poly1 = cs.Polyhedron(A1, b1)
poly2 = cs.Polyhedron(A2, b2)

assert cs.contains(poly1, [1,1])
# TODO distance should be an expression, i.e. norm2(poly1 - poly2)
print cs.dist(poly1, poly2)
elem = cs.proj(poly1, poly2)
assert cs.contains(poly1, elem)
assert cs.dist(poly1, elem) < 1e-6

hull = cs.ConvexHull([b1, b2])
print cs.contains(hull, b1)
print cs.contains(hull, 0.3*b1 + 0.7*b2)

print cs.dist(poly1, 5*hull[0:2] + 2)
print cs.dist(poly1, np.matrix("1 5; -1 3")*poly2 + [1,5])
d1 = cs.dist(poly1, np.matrix("1 0; 0 1")*poly2 + [1,5])
d2 = cs.dist(poly2, poly1 - [1,5])
assert abs(d1 - d2) < 1e-6

poly_hull = hull[0:2] + poly1 + poly2
assert cs.dist(poly_hull, poly1) > 0
intersection = cs.intersect(poly_hull, poly1)
assert cs.is_empty(intersection)
assert not cs.is_empty(poly1)
