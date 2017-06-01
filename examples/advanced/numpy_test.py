"""
Copyright 2013 Steven Diamond

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

import numpy
import abc
class Meta(object):
    def __subclasscheck__(cls, subclass):
        print "hello"

    def __array_finalize__(self, obj):
        return 1

class Test(numpy.ndarray):
    def __init__(self, shape):
        pass

    def __coerce__(self, other):
        print other
        return (self,self)

    def __radd__(self, other):
        print other

    def __getattribute__(self, name):
        import pdb; pdb.set_trace()
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError("'Test' object has no attribute 'affa'")

print issubclass(Test, Meta)
print issubclass(Meta, numpy.ndarray)
print issubclass(Test, numpy.ndarray)
print issubclass(numpy.ndarray, Test)

a = numpy.arange(2)
t = Test(1)
a + t
import pdb; pdb.set_trace()
