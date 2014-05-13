"""Taken from http://code.activestate.com/recipes/576694/
"""

import collections
import itertools

class OrderedSet(collections.MutableSet):
    """A set with ordered keys.

    Backed by a map and linked list.
    """

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        """Adds the key to the set.

        Args:
            key: A hashable object.
        """
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def concat(self, other):
        """Concatenates two ordered sets.

        Args:
            other: An OrderedSet

        Returns:
            An OrderedSet with self's keys followed by other's keys.
        """
        return OrderedSet(itertools.chain(self, other))

    def discard(self, key):
        """Removes the key from the set.

        Preserves the order of the remaining keys.

        Args:
            key: A hashable object.
        """
        if key in self.map:
            key, prev, next_ = self.map.pop(key)
            prev[2] = next_
            next_[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        """Adds the key to the set.

        Args:
            last: If True returns the last element. If False returns the first.

        Returns:
            The last (or first) element in the set.
        """
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)
