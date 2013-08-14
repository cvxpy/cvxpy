from random import random
from cvxpy import *

# An object oriented max-flow problem.
class Edge(object):
    """ An undirected, capacity limited edge. """
    def __init__(self, capacity):
        self.capacity = capacity
        self.flow = Variable()

    def constraints(self):
        return [abs(self.flow) <= self.capacity]

class Node(object):
    """ A node with no accumulation. """
    def __init__(self, edges=[]):
        self.edges = edges
    
    def constraints(self):
        return [sum(e.flow for e in self.edges) == 0]

# Construct a random graph.
N = 20
p = 0.2
nodes = [Node() for i in range(N)]
edges = []
for i in range(N):
    for j in range(i,N):
        # Connect nodes with probability p.
        if random() < p:
            edges.append(Edge(1))
            nodes[i].edges.append(edges[-1])
            nodes[j].edges.append(edges[-1])
# Add a source and sink.
source = Edge(Variable())
nodes[0].edges.append(source)
sink = Edge(Variable())
nodes[-1].edges.append(sink)
edges += [source, sink]

# Construct the problem.
constraints = []
map(constraints.extend, (o.constraints() for o in nodes + edges))
p = Problem(Maximize(source.flow), constraints)
result = p.solve()
print result