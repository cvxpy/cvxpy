from cvxpy import *
import create_graph as g
import pickle

# An object oriented max-flow problem.
class Edge(object):
    """ A directed, capacity limited edge. """
    def __init__(self, capacity):
        self.capacity = capacity
        self.flow = Variable()

    def constraints(self):
        return [0 <= self.flow, self.flow <= self.capacity]

class Node(object):
    """ A node with no accumulation. """
    def __init__(self):
        self.in_edges = []
        self.out_edges = []
    
    def constraints(self):
        in_flow = sum(e.flow for e in self.in_edges)
        out_flow = sum(e.flow for e in self.out_edges)
        return [in_flow == out_flow]

# Read a graph from a file.
f = open(g.FILE, 'r')
data = pickle.load(f)
f.close()

# Construct nodes and edges.
nodes = [Node() for i in range(data[g.NODE_COUNT_KEY])]
edges = []
for n1,n2,capacity in data[g.EDGES_KEY]:
    edges.append(Edge(capacity))
    nodes[n1].out_edges.append(edges[-1])
    nodes[n2].in_edges.append(edges[-1])

# Add source and sink.
source = Edge(Variable())
nodes[0].in_edges.append(source)
sink = Edge(Variable())
nodes[-1].out_edges.append(sink)
edges += [source, sink]

# Construct the problem.
constraints = []
map(constraints.extend, (o.constraints() for o in nodes + edges))
p = Problem(Maximize(source.flow), constraints)
result = p.solve()
print result