"""
Copyright 2025 CVXPY developers  

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


class SolverInfo:
    """A context class that propagates solver attributes 
    through the solving chain.
    """
    def __init__(self, solver=None, supported_constraints=None):
        self.solver_name = solver
        self.solver_supported_constraints = supported_constraints