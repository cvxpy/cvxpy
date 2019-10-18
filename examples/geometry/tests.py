"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Problems involving polyhedra.

from .convex_sets import ConvexHull, Polyhedron, intersect, is_empty, dist, proj, contains
import numpy as np

n = 2
m = 2*n
A1 = np.matrix("1 1; 1 -1; -1 1; -1 -1")
A2 = np.matrix("1 0; -1 0; 0 1; 0 -1")
b1 = 2*np.ones((m,1))
b2 = np.matrix("5; -3; 4; -2")

poly1 = Polyhedron(A1, b1)
poly2 = Polyhedron(A2, b2)

assert contains(poly1, [1,1])
# TODO distance should be an expression, i.e. norm2(poly1 - poly2)
print(dist(poly1, poly2))
elem = proj(poly1, poly2)
assert contains(poly1, elem)
assert dist(poly1, elem) < 1e-6

hull = ConvexHull([b1, b2])
print(contains(hull, b1))
print(contains(hull, 0.3*b1 + 0.7*b2))

print(dist(poly1, 5*hull[0:2] + 2))
print(dist(poly1, np.matrix("1 5; -1 3") * poly2 + [1,5]))
d1 = dist(poly1, np.matrix("1 0; 0 1")*poly2 + [1,5])
d2 = dist(poly2, poly1 - [1,5])
assert abs(d1 - d2) < 1e-6

poly_hull = hull[0:2] + poly1 + poly2
assert dist(poly_hull, poly1) > 0
intersection = intersect(poly_hull, poly1)
assert is_empty(intersection)
assert not is_empty(poly1)
