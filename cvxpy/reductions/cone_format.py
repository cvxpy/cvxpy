"""
Copyright, the CVXPY authors

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
from cvxpy.reductions.reduction import Reduction


class ConeFormat(Reduction):
    """Apply a conic solver's cone row layout as an explicit chain step.

    This is the only place cone formatting happens. Interfaces used to each
    open by calling ``format_constraints`` themselves, guarded on
    ``problem.formatted`` -- eleven copies of a step every new interface had
    to remember, and one that ``Dualize`` already assumed had run. An
    already-formatted program is now a precondition of ``ConicSolver.apply``
    rather than something each interface repairs.

    Making it a reduction also lets a program that owns a re-extractable form
    restructure itself instead of being rebuilt as an ordinary program, via
    ``ParamConeProg.format_for``.

    A no-op when there is nothing to do: an already-formatted program (which
    is what ``ExtractDirectCones`` leaves behind), or a program whose solver
    formats nothing.
    """

    def __init__(self, solver) -> None:
        super().__init__()
        self.solver = solver

    def accepts(self, problem) -> bool:
        return True

    def apply(self, problem):
        if getattr(problem, 'formatted', True):
            return problem, None
        return problem.format_for(self.solver), None

    def invert(self, solution, inverse_data):
        return solution
