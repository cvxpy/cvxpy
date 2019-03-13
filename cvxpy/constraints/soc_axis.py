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

from cvxpy.constraints.second_order import SOC


class SOC_Axis(SOC):
    """A second-order cone constraint for each row/column.

    Assumes t is a vector the same length as X's columns (rows) for axis==0 (1).

    Attributes:
        t: The scalar part of the second-order constraint.
        X: A matrix whose rows/columns are each a cone.
        axis: Slice by column 0 or row 1.
    """

    def __init__(self, t, X, axis):
        assert t.shape[1] == 1
        self.t = t
        self.x_elems = [X]
        self.axis = axis
        super(SOC_Axis, self).__init__(t, X)

    def __str__(self):
        return "SOC_Axis(%s, %s, %s)" % (self.t, self.X, self.axis)

    def num_cones(self):
        """The number of elementwise cones.
        """
        return self.t.shape[0]

    def cone_size(self):
        """The dimensions of a single cone.
        """
        return (1 + self.x_elems[0].shape[self.axis], 1)

    @property
    def size(self):
        """The dimensions of the second-order cones.

        Returns
        -------
        list
            A list of the dimensions of the elementwise cones.
        """
        cones = []
        cone_size = self.cone_size()
        for i in range(self.num_cones()):
            cones.append(cone_size)
        return cones
