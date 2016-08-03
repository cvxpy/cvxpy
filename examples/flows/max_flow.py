from cvxpy import *
import create_graph as g
import pickle

# An object oriented max-flow problem.
class Edge(object):
    """ An undirected, capacity limited edge. """
    def __init__(self, capacity):
        self.capacity = capacity
        self.flow = Variable()

    # Connects two nodes via the edge.
    def connect(self, in_node, out_node):
        in_node.edge_flows.append(-self.flow)
        out_node.edge_flows.append(self.flow)

    # Returns the edge's internal constraints.
    def constraints(self):
        return [abs(self.flow) <= self.capacity]

class Node(object):
    """ A node with accumulation. """
    def __init__(self, accumulation=0):
        self.accumulation = accumulation
        self.edge_flows = []

    # Returns the node's internal constraints.
    def constraints(self):
        return [sum(f for f in self.edge_flows) == self.accumulation]

if __name__ == "__main__":
    # Read a graph from a file.
    f = open(g.FILE, 'r')
    data = pickle.load(f)
    f.close()

    # Construct nodes.
    node_count = data[g.NODE_COUNT_KEY]
    nodes = [Node() for i in range(node_count)]
    # Add source.
    nodes[0].accumulation = Variable()
    # Add sink.
    nodes[-1].accumulation = Variable()

    # Construct edges.
    edges = []
    for n1,n2,capacity in data[g.EDGES_KEY]:
        edges.append(Edge(capacity))
        edges[-1].connect(nodes[n1], nodes[n2])
    # Construct the problem.
    constraints = []
    for o in nodes + edges:
        constraints += o.constraints()
    p = Problem(Maximize(nodes[-1].accumulation), constraints)
    result = p.solve()
    print result
