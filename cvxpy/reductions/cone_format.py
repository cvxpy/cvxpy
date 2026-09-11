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
from cvxpy.problems.param_prob import ParamProb
from cvxpy.reductions.reduction import Reduction


class ConeFormat(Reduction):
    """Apply a conic solver's cone row layout as an explicit chain step.

    Interfaces used to each open by calling ``format_constraints``
    themselves, guarded on ``problem.formatted`` -- eleven copies of a step
    every new interface had to remember, and one that ``Dualize`` already
    assumed had run. An already-formatted program is now a precondition of
    ``ConicSolver.apply`` rather than something each interface repairs.

    Which formatting to apply is dispatched on the program, via
    ``format_for(solver)``, so a program that knows a cheaper way to
    restructure itself can say so rather than be rebuilt.

    ``ExtractDirectCones`` is the one reduction that still formats on its own
    (``cone2cone/extract_direct_cones.py``). It has to: it scans rows in
    per-cone order to find its cones, which is earlier than this step runs.
    It leaves ``formatted`` set, so this reduction is then a no-op.
    """

    def __init__(self, solver) -> None:
        super().__init__()
        self.solver = solver

    def accepts(self, problem) -> bool:
        return isinstance(problem, ParamProb)

    def apply(self, problem):
        if problem.formatted:
            return problem, None
        return problem.format_for(self.solver), None

    def invert(self, solution, inverse_data):
        return solution
