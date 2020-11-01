from cvxpy.settings import CPLEX


def _refine_conflicts_if_infeasible(soln):
    """
    Run Cplex's conflict refiner if the solution is infeasible and print the results.

    TODO: This function is very limited as the documentation of conflict refiner is 
        brief and does not contain any examples. This will catch only some basic cases.
        If you know how to use it properly, feel free to iterate on this.
    """
    c = soln['model']
    solution_status = c.solution.status[c.solution.get_status()]
    infeasible_statuses = [
        'infeasible', 'MIP_infeasible', 'infeasible_or_unbounded', 'optimal_infeasible'
    ]
    if solution_status not in infeasible_statuses:
        return

    print("Starting solution refiner.")
    c.conflict.refine(c.conflict.all_constraints())
    print("Solution refiner ended.")

    def var_name(idx):
        try:
            return c.variables.get_names(idx)
        except:
            return f"x{idx}"

    def linear_constraint_name(idx):
        try:
            return c.linear_constraints.get_names(idx)
        except:
            return f"c{idx}"

    conflict_groups = c.conflict.get_groups()
    group_status_inds = c.conflict.get()
    for i, (_, constrs) in enumerate(conflict_groups):
        group_status_ind = group_status_inds[i]
        group_status = c.conflict.group_status[group_status_ind]
        if group_status == 'excluded':
            continue
        print(f"CONFLICT GROUP: {group_status}")
        for constr_type, constr_id in constrs:
            if constr_type == c.conflict.constraint_type.linear:
                constraint_name = linear_constraint_name(constr_id)
                lhs = c.linear_constraints.get_rows(constr_id)
                rhs = c.linear_constraints.get_rhs(constr_id)
                sense = c.linear_constraints.get_senses(constr_id)
                sense_dict = {'E': '=', 'G': '≥', 'L': '≤'}
                lhs_str = " + ".join([f"{val}*{var_name(ind)}" for ind, val in zip(lhs.ind, lhs.val)])
                print(f"Linear constraint: {constraint_name}:\n    {lhs_str} {sense_dict[sense]} {rhs}")
            elif constr_type == c.conflict.constraint_type.upper_bound:
                ub = c.variables.get_upper_bounds(constr_id)
                var_name = var_name(constr_id)
                print(f"Upper bound:\n    {var_name} ≤ {ub}")
            elif constr_type == c.conflict.constraint_type.lower_bound:
                lb = c.variables.get_lower_bounds(constr_id)
                var_name = var_name(constr_id)
                print(f"Lower bound:\n    {var_name} ≥ {lb}")
            else:
                raise Exception(f"Unknown constraint type {constr_type}.")
        print()


def solve_with_conflict_refiner(problem, warm_start=False, verbose=False, solver_opts={}, gp=False, enforce_dpp=False):
    """
    Solve the problem using CPLEX. If the solution is infeasible use the conflict refiner
    and print the results. Here is an example:

        problem = cp.Problem(objective, constraints.get())
        cp.cplex.solve_with_conflict_refiner(problem)

    Arguments
        ---------
        warm_start : bool, optional
            Value is passed through to the `chain.solve_via_data`.
        verbose : bool, optional
            Value is passed through to the `chain.solve_via_data`.
        solver_opts : dict, optional
            Value is passed through to the `chain.solve_via_data`.
        gp : bool, optional
            If True, then parses the problem as a disciplined geometric program
            instead of a disciplined convex program.
        enforce_dpp : bool, optional
            When True, a DPPError will be thrown when trying to parse a non-DPP
            problem (instead of just a warning). Defaults to False.
    """
    data, chain, inverse_data = problem.get_problem_data(solver=CPLEX, gp=gp, enforce_dpp=enforce_dpp)
    soln = chain.solve_via_data(problem, data, warm_start, verbose, solver_opts)
    problem.unpack_results(soln, chain, inverse_data)
    _refine_conflicts_if_infeasible(soln)
