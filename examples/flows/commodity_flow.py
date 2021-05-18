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

from cvxpy import *
import create_graph as g
from max_flow import Edge, Node
import pickle
import random as r
import cvxopt

# Multi-commodity flow.
COMMODITIES = 5 # Number of commodities.
r.seed(1)

class MultiEdge(Edge):
    """ An undirected, capacity limited edge with multiple commodities. """
    def __init__(self, capacity):
        self.capacity = capacity
        self.in_flow = Variable(COMMODITIES)
        self.out_flow = Variable(COMMODITIES)

    # Returns the edge's internal constraints.
    def constraints(self):
        return [self.in_flow + self.out_flow == 0,
                sum(abs(self.in_flow)) <= self.capacity]

class MultiNode(Node):
    """ A node with a target flow accumulation and a capacity. """
    def __init__(self, capacity: float = 0.0):
        self.capacity = capacity
        self.edge_flows = []

    # The total accumulation of flow.
    def accumulation(self):
        return sum(f for f in self.edge_flows)

    def constraints(self):
        return [abs(self.accumulation()) <= self.capacity]


if __name__ == "__main__":
    # Read a graph from a file.
    f = open(g.FILE, 'r')
    data = pickle.load(f)
    f.close()

    # Construct nodes.
    node_count = data[g.NODE_COUNT_KEY]
    nodes = [MultiNode() for i in range(node_count)]
    # Add a source and sink for each commodity.
    sources = []
    sinks = []
    for i in range(COMMODITIES):
        source,sink = r.sample(nodes, 2)
        # Only count accumulation of a single commodity.
        commodity_vec = cvxopt.matrix(0,(COMMODITIES,1))
        commodity_vec[i] = 1
        # Initialize the source.
        source.capacity = commodity_vec*Variable()
        sources.append(source)
        # Initialize the sink.
        sink.capacity = commodity_vec*Variable()
        sinks.append(sink)

    # Construct edges.
    edges = []
    for n1,n2,capacity in data[g.EDGES_KEY]:
        edges.append(MultiEdge(capacity))
        edges[-1].connect(nodes[n1], nodes[n2])

    # Construct the problem.
    objective = Maximize(sum(sum(s.accumulation() for s in sinks)))
    constraints = []
    for o in nodes + edges:
        constraints += o.constraints()
    p = Problem(objective, constraints)
    result = p.solve()
    print("Objective value = %s" % result)
    # Show how the flow for each commodity.
    for i,s in enumerate(sinks):
        accumulation = sum(f.value[i] for f in s.edge_flows)
        print("Accumulation of commodity %s = %s" % (i, accumulation))
