"""
Copyright 2016 Jaehyun Park, 2017 Robin Verschueren

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""


class InverseData(object):
    """ TODO(akshayka): Document this class."""

    def __init__(self, problem):
        obj, cons = problem.objective, problem.constraints
        all_variables = obj.variables() + [var for c in cons
                                           for var in c.variables()]
        varis = list(set(all_variables))
        self.id_map, self.var_offsets, self.x_length, self.var_shapes = (
                                                self.get_var_offsets(varis))
        self.id2var = {id: var.id for var in varis}
        self.cons_id_map = dict()

    def get_var_offsets(self, variables):
        var_shapes = {}
        var_offsets = {}
        id_map = {}
        vert_offset = 0
        for x in variables:
            var_shapes[x.id] = x.shape
            var_offsets[x.id] = vert_offset
            id_map[x.id] = (vert_offset, x.size)
            vert_offset += x.size
        return (id_map, var_offsets, vert_offset, var_shapes)
