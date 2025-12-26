"""
Copyright 2013 Steven Diamond, 2017 Akshay Agrawal

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


from cvxpy.reductions.inverse_data import InverseData


class SolverInverseData(InverseData):
    """
    The context for a reduction, which may include a solver and solver options.
    """
    def __init__(self, inverse_data: dict, solver_instance = None,
                 solver_options: dict = {}):
        self.solver_instance = solver_instance
        self.solver_options = solver_options
        self.inverse_data = inverse_data

    def __getitem__(self, key):
        """Get a value from inverse_data using dictionary-like syntax.

        Args:
            key: The key to retrieve from inverse_data.

        Returns:
            The value associated with the key.

        Raises:
            KeyError: If the key doesn't exist in inverse_data.
        """
        return self.inverse_data[key]

    def get(self, key, default=None):
        """Get a value from inverse_data, returning default if not found.

        This method provides dictionary-like access compatible with code
        that expects dict.get() behavior.

        Args:
            key: The key to retrieve from inverse_data.
            default: The value to return if the key doesn't exist.

        Returns:
            The value if it exists, otherwise the default value.
        """
        if key in self.inverse_data:
            return self.inverse_data[key]
        return default

    def __contains__(self, key):
        """Check if a key exists in inverse_data.

        Args:
            key: The key to check for.

        Returns:
            True if the key exists, False otherwise.
        """
        return key in self.inverse_data