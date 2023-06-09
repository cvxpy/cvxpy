"""
Copyright 2022, the CVXPY authors.

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
from typing import Tuple, Union


class Version:
    """
    Parses strings of the form 'x.y.z+[STUFF]' and tuples of the form (x, y, z).
    In the former case, '+[STUFF]' refers to a local version identifier in the
    sense of https://www.python.org/dev/peps/pep-0440/#local-version-identifiers.
    We don't actually store the local version identifier for comparisons.
    """

    def __init__(self, v: Union[str, Tuple[int, int, int]]):
        if isinstance(v, str):
            v = v.split('rc')[0].split('.')
            assert len(v) >= 3  # anything after the third place doesn't matter
            v[2] = v[2].split('+')[0]  # anything after the + doesn't matter
        self.major = int(v[0])
        self.minor = int(v[1])
        self.micro = int(v[2])
        self.v = (self.major, self.minor, self.micro)

    def __le__(self, other):
        return self.v <= other.v

    def __lt__(self, other):
        return self.v < other.v

    def __ge__(self, other):
        return self.v >= other.v

    def __gt__(self, other):
        return self.v > other.v

    def __eq__(self, other):
        return self.v == other.v

    def __ne__(self, other):
        return self.v != other.v

    def __str__(self):
        return '%s.%s.%s' % self.v
