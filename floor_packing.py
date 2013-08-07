from cvxpy import *
import pylab

class Box(object):
    """ A box in a floor packing problem. """
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.x = Variable()
        self.y = Variable()

    @property
    def position(self):
        return (round(self.x.value - self.width/2,2), 
                round(self.y.value - self.height/2,2))

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
    MARGIN = 1
    def __init__(self, boxes):
        self.boxes = boxes
        self.height = Variable()
        self.width = Variable()
        self.horizontal_orderings = []
        self.vertical_orderings = []

    @property
    def size(self):
        return (round(self.width.value,2), round(self.height.value,2))

    # Return constraints for the ordering.
    @staticmethod
    def _order(boxes, horizontal):
        if len(boxes) == 0: return
        constraints = []
        curr = boxes[0]
        for box in boxes[1:]:
            if horizontal:
                constraints.append(curr.right + FloorPlan.MARGIN <= box.left)
            else:
                constraints.append(curr.top + FloorPlan.MARGIN <= box.bottom)
            curr = box
        return constraints

    # Compute minimum perimeter layout.
    def layout(self):
        constraints = []
        # Enforce that boxes lie in bounding box.
        for box in self.boxes:
            constraints += [box.bottom >= FloorPlan.MARGIN, 
                            box.top + FloorPlan.MARGIN <= self.height]
            constraints += [box.left >= FloorPlan.MARGIN, 
                            box.right + FloorPlan.MARGIN <= self.width]
        # Enforce the relative ordering of the boxes.
        for ordering in self.horizontal_orderings:
            constraints += self._order(ordering, True)
        for ordering in self.vertical_orderings:
            constraints += self._order(ordering, False)
        p = Problem(Minimize(2*(self.height + self.width)), constraints)
        return p.solve()

    # Show the layout with matplotlib
    def show(self):
        pylab.figure(facecolor='w')
        for k in range(len(self.boxes)):
            box = self.boxes[k]
            x,y = box.position
            pylab.fill([x, x, x + box.width, x + box.width],
                       [y, y+box.height, y+box.height, y],
                       facecolor = '#D0D0D0')
            pylab.text(x+.5*box.width, y+.5*box.height, "%d" %(k+1))
        x,y = self.size
        pylab.axis([0, x, 0, y])
        pylab.xticks([])
        pylab.yticks([])

        pylab.show()

boxes = [Box(10,20), Box(5,8), Box(9,3), Box(40,20), Box(9,10)]
fp = FloorPlan(boxes)
fp.horizontal_orderings.append( [boxes[0], boxes[2], boxes[4]] )
fp.horizontal_orderings.append( [boxes[1], boxes[2]] )
fp.vertical_orderings.append( [boxes[4], boxes[3]] )
fp.vertical_orderings.append( [boxes[1], boxes[0], boxes[3]] )
fp.vertical_orderings.append( [boxes[2], boxes[3]] )
fp.layout()
fp.show()