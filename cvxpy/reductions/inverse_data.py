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

import cvxpy.lin_ops.lin_op as lo


class InverseData:
    """Stores data useful for solution retrieval."""

    def __init__(self, problem) -> None:
        varis = problem.variables()
        self.id_map, self.var_offsets, self.x_length, self.var_shapes = \
            InverseData.get_var_offsets(varis)

        self.param_shapes = {}
        # Always start with CONSTANT_ID.
        self.param_to_size = {lo.CONSTANT_ID: 1}
        self.param_id_map = {}
        offset = 0
        for param in problem.parameters():
            self.param_shapes[param.id] = param.shape
            self.param_to_size[param.id] = param.size
            self.param_id_map[param.id] = offset
            offset += param.size
        self.param_id_map[lo.CONSTANT_ID] = offset

        self.id2var = {var.id: var for var in varis}
        self.id2cons = {cons.id: cons for cons in problem.constraints}
        self.cons_id_map = dict()
        self.constraints = None

    @staticmethod
    def get_var_offsets(variables):
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
