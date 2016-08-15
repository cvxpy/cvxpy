from cvxpy import *
import pylab
import math

# Based on http://cvxopt.org/examples/book/floorplan.html
class Box(object):
    """ A box in a floor packing problem. """
    ASPECT_RATIO = 5.0
    def __init__(self, min_area):
        self.min_area = min_area
        self.height = Variable()
        self.width = Variable()
        self.x = Variable()
        self.y = Variable()

    @property
    def position(self):
        return (round(self.x.value,2), round(self.y.value,2))

    @property
    def size(self):
        return (round(self.width.value,2), round(self.height.value,2))

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y

    @property
    def top(self):
        return self.y + self.height

class FloorPlan(object):
    """ A minimum perimeter floor plan. """
    MARGIN = 1.0
    ASPECT_RATIO = 5.0
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
        for box in self.boxes:
            # Enforce that boxes lie in bounding box.
            constraints += [box.bottom >= FloorPlan.MARGIN,
                            box.top + FloorPlan.MARGIN <= self.height]
            constraints += [box.left >= FloorPlan.MARGIN,
                            box.right + FloorPlan.MARGIN <= self.width]
            # Enforce aspect ratios.
            constraints += [(1/box.ASPECT_RATIO)*box.height <= box.width,
                            box.width <= box.ASPECT_RATIO*box.height]
            # Enforce minimum area
            constraints += [
            geo_mean(vstack(box.width, box.height)) >= math.sqrt(box.min_area)
            ]

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
            w,h = box.size
            pylab.fill([x, x, x + w, x + w],
                       [y, y+h, y+h, y],
                       facecolor = '#D0D0D0')
            pylab.text(x+.5*w, y+.5*h, "%d" %(k+1))
        x,y = self.size
        pylab.axis([0, x, 0, y])
        pylab.xticks([])
        pylab.yticks([])

        pylab.show()

boxes = [Box(180), Box(80), Box(80), Box(80), Box(80)]
fp = FloorPlan(boxes)
fp.horizontal_orderings.append( [boxes[0], boxes[2], boxes[4]] )
fp.horizontal_orderings.append( [boxes[1], boxes[2]] )
fp.horizontal_orderings.append( [boxes[3], boxes[4]] )
fp.vertical_orderings.append( [boxes[1], boxes[0], boxes[3]] )
fp.vertical_orderings.append( [boxes[2], boxes[3]] )
fp.layout()
fp.show()
