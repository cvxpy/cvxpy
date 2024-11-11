
Object Oriented Convex Optimization
===================================

CVXPY enables an object-oriented approach to constructing optimization
problems. An object-oriented approach is simpler and more flexible than
the traditional method of constructing problems by embedding information
in matrices.

Consider the max-flow problem with ``N`` nodes and ``E`` edges. We can
define the problem explicitly by constructing an ``N`` by ``E``
incidence matrix ``A``. ``A[i, j]`` is +1 if edge ``j`` enters node
``i``, -1 if edge ``j`` leaves node ``i``, and 0 otherwise. The source
and sink are the last two edges. The problem becomes

.. code:: python

    # A is the incidence matrix. 
    # c is a vector of edge capacities.
    flows = Variable(E-2)
    source = Variable()
    sink = Variable()
    p = Problem(Maximize(source),
                  [A*hstack([flows,source,sink]) == 0,
                   0 <= flows,
                   flows <= c])

The more natural way to frame the max-flow problem is not in terms of
incidence matrices, however, but in terms of the properties of edges and
nodes. We can write an ``Edge`` class to capture these properties.

.. code:: python

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

The ``Edge`` class exposes the flow into and out of the edge. The
capacity constraint is stored locally in the ``Edge`` object. The graph
structure is also stored locally, by calling
``edge.connect(node1, node2)`` for each edge.

We also define a ``Node`` class:

.. code:: python

    class Node(object):
        """ A node with accumulation. """
        def __init__(self, accumulation=0):
            self.accumulation = accumulation
            self.edge_flows = []
    
        # Returns the node's internal constraints.
        def constraints(self):
            return [sum(f for f in self.edge_flows) == self.accumulation]

Nodes have a target amount of flow to accumulate. Sources and sinks are
Nodes with a variable as their accumulation target.

Suppose ``nodes`` is a list of all the nodes, ``edges`` is a list of all
the edges, and ``sink`` is the sink node. The problem becomes:

.. code:: python

    constraints = []
    for obj in nodes + edges:
        constraints += obj.constraints()
    prob = Problem(Maximize(sink.accumulation), constraints)

Note that the problem has been reframed from maximizing the flow along
the source edge to maximizing the accumulation at the sink node. We
could easily extend the ``Edge`` and ``Node`` class to model an
electrical grid. Sink nodes would be consumers. Source nodes would be
power stations, which generate electricity at a cost. A node could be
both a source and a sink, which would represent energy storage
facilities or a consumer who contributes to the grid. We could add
energy loss along edges to more accurately model transmission lines. The
entire grid construct could be embedded in a time series model.

To see the object-oriented approach applied to more complex flow
problems, look in the ``cvxpy/examples/flows/`` directory.
