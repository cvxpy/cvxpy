from cvxpy import settings as s
from cvxpy.reductions.reduction import Reduction


def _compose_id_map(step_maps):
    """Compose a sequence of ``{old_id: [new_id, ...]}`` mappings.

    Each step map comes from a reduction's ``var_id_map`` or
    ``param_id_map``.  The result is a single mapping from the
    outermost original IDs to lists of the innermost final IDs.
    """
    result = {}
    for step_map in step_maps:
        # Update existing mappings whose current targets were renamed.
        for orig_id, cur_ids in result.items():
            new_ids = []
            for cur_id in cur_ids:
                if cur_id in step_map:
                    new_ids.extend(step_map[cur_id])
                else:
                    new_ids.append(cur_id)
            result[orig_id] = new_ids
        # Add new mappings introduced by this reduction step.
        for orig_id in step_map.keys() - result.keys():
            result[orig_id] = list(step_map[orig_id])
    return result


class Chain(Reduction):
    """A logical grouping of multiple reductions into a single reduction.

    Attributes
    ----------
    reductions : list[Reduction]
        A list of reductions.
    """

    def __init__(self, problem=None, reductions=None) -> None:
        super(Chain, self).__init__(problem=problem)
        self.reductions = [] if reductions is None else reductions

    def __str__(self):
        return str(self.reductions)

    def __repr__(self) -> str:
        return "Chain(reductions=%s)" % repr(self.reductions)

    def get(self, reduction_type):
        for reduction in self.reductions:
            if isinstance(reduction, reduction_type):
                return reduction
        raise KeyError

    def accepts(self, problem) -> bool:
        """A problem is accepted if the sequence of reductions is valid.

        In particular, the i-th reduction must accept the output of the i-1th
        reduction, with the first reduction (self.reductions[0])
        in the sequence taking as input the supplied problem.

        Parameters
        ----------
        problem : Problem
            The problem to check.

        Returns
        -------
        bool
            True if the chain can be applied, False otherwise.
        """

        for r in self.reductions:
            if not r.accepts(problem):
                return False
            problem, _ = r.apply(problem)
        return True

    def apply(self, problem, verbose: bool = False):
        """Applies the chain to a problem and returns an equivalent problem.

        Parameters
        ----------
        problem : Problem
            The problem to which the chain will be applied.
        verbose : bool, optional
            Whehter to print verbose output.

        Returns
        -------
        Problem or dict
            The problem yielded by applying the reductions in sequence,
            starting at self.reductions[0].
        list
            The inverse data yielded by each of the reductions.
        """
        inverse_data = []
        for r in self.reductions:
            if verbose:
                s.LOGGER.info('Applying reduction %s', type(r).__name__)
            problem, inv = r.apply(problem)
            inverse_data.append(inv)
        return problem, inverse_data

    def compose_var_id_map(self):
        """Compose variable ID mappings across all reductions.

        Returns a single ``{orig_var_id: [final_var_id, ...]}`` dict that
        accounts for every variable replacement in the chain.  If
        reduction *i* maps ``A → [A']`` and reduction *j* (j > i) maps
        ``A' → [A'']``, the result contains ``A → [A'']``.  If a step
        maps ``A' → [A'', A''']``, the result expands to
        ``A → [A'', A''']``.

        Returns
        -------
        dict
            Maps original variable IDs to lists of their final (innermost)
            IDs.
        """
        return _compose_id_map(r.var_id_map for r in self.reductions)

    def compose_param_id_map(self):
        """Compose parameter ID mappings across all reductions.

        Returns a single ``{orig_param_id: [final_param_id, ...]}`` dict
        that accounts for every parameter replacement in the chain.

        Returns
        -------
        dict
            Maps original parameter IDs to lists of their final (innermost)
            IDs.
        """
        return _compose_id_map(r.param_id_map for r in self.reductions)

    def invert(self, solution, inverse_data):
        """Returns a solution to the original problem given the inverse_data.
        """
        for r, inv in reversed(list(zip(self.reductions, inverse_data))):
            solution = r.invert(solution, inv)
        return solution
