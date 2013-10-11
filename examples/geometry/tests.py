# Problems involving polyhedra.

from set_variables import Polyhedron, ConvexHull
from cvxpy import numpy as np
import cvxpy

n = 2
m = 2*n
A1 = np.matrix("1 1; 1 -1; -1 1; -1 -1")
A2 = np.matrix("1 0; -1 0; 0 1; 0 -1")
b1 = 2*np.ones((m,1))
b2 = np.matrix("5; -3; 4; -2")

poly1 = Polyhedron(A1, b1)
poly2 = Polyhedron(A2, b2)

print poly1.contains([1,1])
print poly1.dist(poly2)
elem = poly1.proj(poly2)
print poly1.contains(elem)
print poly2.dist(elem)

hull = ConvexHull([b1, b2])
print hull.contains(b1)
print hull.contains(0.3*b1 + 0.7*b2)

print poly1.dist(5*hull[0:2] + 2)
print poly1.dist(np.matrix("1 5; -1 3")*poly2 + [1,5])
print poly1.dist(np.matrix("1 0; 0 1")*poly2 + [1,5]) - poly2.dist(poly1 - [1,5])

poly_hull = hull[0:2] + poly1 + poly2
print poly1.dist(poly_hull)
intersect = poly1.intersect(poly2)
print intersect.is_empty()
print poly1.is_empty()