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
