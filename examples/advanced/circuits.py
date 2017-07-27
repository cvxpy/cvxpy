# An object oriented model of a circuit.
from cvxpy import *
import abc

class Node(object):
    """ A node connecting devices. """
    def __init__(self):
        self.voltage = Variable()
        self.current_flows = []

    # The current entering a node equals the current leaving the node.
    def constraints(self):
        return [sum(f for f in self.current_flows) == 0]

class Ground(Node):
    """ A node at 0 volts. """
    def constraints(self):
        return [self.voltage == 0] + super(Ground, self).constraints()
    
class Device(object):
    __metaclass__ = abc.ABCMeta
    """ A device on a circuit. """
    def __init__(self, pos_node, neg_node):
        self.pos_node = pos_node
        self.pos_node.current_flows.append(-self.current())
        self.neg_node = neg_node
        self.neg_node.current_flows.append(self.current())

    # The voltage drop on the device.
    @abc.abstractmethod
    def voltage(self):
        return NotImplemented

    # The current through the device.
    @abc.abstractmethod
    def current(self):
        return NotImplemented

    # Every path between two nodes has the same voltage drop.
    def constraints(self):
        return [self.pos_node.voltage - self.voltage() == self.neg_node.voltage]

class Resistor(Device):
    """ A resistor with V = R*I. """
    def __init__(self, pos_node, neg_node, resistance):
        self._current = Variable()
        self.resistance = resistance
        super(Resistor, self).__init__(pos_node, neg_node)

    def voltage(self):
        return -self.resistance*self.current()

    def current(self):
        return self._current

class VoltageSource(Device):
    """ A constant source of voltage. """
    def __init__(self, pos_node, neg_node, voltage):
        self._current = Variable()
        self._voltage = voltage
        super(VoltageSource, self).__init__(pos_node, neg_node)

    def voltage(self):
        return self._voltage

    def current(self):
        return self._current

class CurrentSource(Device):
    """ A constant source of current. """
    def __init__(self, pos_node, neg_node, current):
        self._current = current
        self._voltage = Variable()
        super(CurrentSource, self).__init__(pos_node, neg_node)

    def voltage(self):
        return self._voltage

    def current(self):
        return self._current

# # Create a simple circuit and find the current and voltage.
nodes = [Ground(),Node(),Node()]
# A 5 V battery
devices = [VoltageSource(nodes[0], nodes[2], 10)]
# A series of pairs of parallel resistors.
# 1/4 Ohm resistor and a 1 Ohm resistor in parallel.
devices.append( Resistor(nodes[0], nodes[1], 0.25) )
devices.append( Resistor(nodes[0], nodes[1], 1) )
# 4 Ohm resistor and a 1 Ohm resistor in parallel.
devices.append( Resistor(nodes[1], nodes[2], 4) )
devices.append( Resistor(nodes[1], nodes[2], 1) )

# Create the problem.
constraints = []
for obj in nodes + devices:
    constraints += obj.constraints()
Problem(Minimize(0), constraints).solve()
for node in nodes:
    print node.voltage.value
