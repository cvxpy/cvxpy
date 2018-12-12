"""
Copyright 2016 Jaehyun Park, 2017 Robin Verschueren

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

import cvxpy.lin_ops.lin_utils as lu


class InverseData(object):
    """ TODO(akshayka): Document this class."""

    def __init__(self, problem):
        varis = problem.variables()
        self.id_map, self.var_offsets, self.x_length, self.var_shapes = (
                                                self.get_var_offsets(varis))
        self.id2var = {var.id: var for var in varis}
        self.real2imag = {var.id: lu.get_id() for var in varis
                          if var.is_complex()}
        constr_dict = {cons.id: lu.get_id() for cons in problem.constraints
                       if cons.is_complex()}
        self.real2imag.update(constr_dict)
        self.id2cons = {cons.id: cons for cons in problem.constraints}
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
