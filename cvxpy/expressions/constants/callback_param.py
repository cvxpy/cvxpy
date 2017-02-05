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

from cvxpy.expressions.constants.parameter import Parameter


class CallbackParam(Parameter):
    """
    A parameter whose value is obtained by evaluating a function.
    """
    PARAM_COUNT = 0

    def __init__(self, atom, rows=1, cols=1, name=None, sign="unknown"):
        self.atom = atom
        super(CallbackParam, self).__init__(rows, cols, name, sign)

    @property
    def value(self):
        """Evaluate the callback to get the value.
        """
        return self._validate_value(self.atom.value)

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self.atom, self._rows, self._cols,
                self._name, self.sign_str]
