"""
Copyright 2017 Steven Diamond

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
import abc


class Constraint(object):
    """Abstract super class for constraints.

    TODO rationalize constraint classes. Make lin_op versions
    of SOC, SDP, etc.

    Attributes
    ----------
    constr_id : int
        A unique id for the constraint.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, constr_id=None):
        if constr_id is None:
            self.constr_id = lu.get_id()
        else:
            self.constr_id = constr_id
        super(Constraint, self).__init__()
