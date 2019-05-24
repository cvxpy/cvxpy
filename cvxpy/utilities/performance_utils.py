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
import functools


def lazyprop(func):
    """Wraps a property so it is lazily evaluated."""
    attr_name = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def _lazyprop(self):
        try:
            return getattr(self, attr_name)
        except AttributeError:
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    return _lazyprop


def compute_once(func):
    """Computes a method exactly once and caches the result."""
    attr_name = '_compute_once_' + func.__name__

    @functools.wraps(func)
    def _compute_once(self):
        try:
            return getattr(self, attr_name)
        except AttributeError:
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    return _compute_once
