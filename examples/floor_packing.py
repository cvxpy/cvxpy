from cvxpy import *

class Box(object):
    """ A box in a floor packing problem. """
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.x = Variable()
        self.y = Variable()

    @property
    def center(self):
        return (self.x.value, self.y.value)

    @property
    def left(self):
        return self.x - self.width/2

    @property
    def right(self):
        return self.x + self.width/2

    @property
    def bottom(self):
        return self.y - self.height/2

    @property
    def top(self):
        return self.y + self.height/2

class FloorPlan(object):
    """ A minimum perimeter floor plan. """
    def __init__(self, boxes):
        self.boxes = boxes
        self.height = Variable()
        self.width = Variable()
        self._order_constraints = []

    @property
    def size(self):
        return (self.height.value, self.width.value)

    # Ensure the boxes are ordered from left to right.
    def order_horizontally(self, boxes):
        if len(boxes) == 0: return
        curr = boxes[0]
        for box in boxes[1:]:
            self._order_constraints.append(curr.right <= box.left)
            curr = box

    # Ensure the boxes are ordered from bottom to top.
    def order_vertically(self, boxes):
        if len(boxes) == 0: return
        curr = boxes[0]
        for box in boxes[1:]:
            self._order_constraints.append(curr.top <= box.bottom)
            curr = box

    # Compute minimum perimeter layout.
    def layout(self):
        constraints = self._order_constraints[:]
        for box in boxes:
            constraints += [box.bottom >= 0, box.top <= self.height]
            constraints += [box.left >= 0, box.right <= self.width]
        p = Problem(Minimize(2*(self.height + self.width)), constraints)
        return p.solve()

box = Box(10,20)
fp = FloorPlan([box])
fp.layout()
print box.center
print fp.size