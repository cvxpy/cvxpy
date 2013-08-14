from cvxpy import *
import create_graph as g
import pickle

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
    def __init__(self):
        self.edges = []
    
    def constraints(self):
        return [sum(e.flow for e in self.edges) == 0]

# Read a graph from a file.
f = open(g.FILE, 'r')
data = pickle.load(f)
f.close()
# Construct nodes and edges.
nodes = [Node() for i in range(data[g.NODE_COUNT_KEY])]
edges = []
for n1,n2,capacity in data[g.EDGES_KEY]:
    edges.append(Edge(capacity))
    nodes[n1].edges.append(edges[-1])
    nodes[n2].edges.append(edges[-1])
# Add source and sink.
source = Edge(Variable())
nodes[0].edges.append(source)
sink = Edge(Variable())
nodes[-1].edges.append(sink)

# Construct the problem.
constraints = [source.flow >= 0, sink.flow >= 0]
map(constraints.extend, (o.constraints() for o in nodes + edges))
p = Problem(Maximize(source.flow), constraints)
result = p.solve()
print result