"""Optimize a design according to preferences."""

import scipy.optimize

from physprog import classfunctions, objective, plots


def optimize(model, preferences, plot=False):
    """Optimize the given problem to specified preferences."""
    constraints = get_constraints(model, preferences)
    aggregate = objective.build_objective(model, preferences)
    initial_performance = model.evaluate()
    val = aggregate(model.design)
    print(f"Optimizing design starting at value: {val:.2f}")
    print(initial_performance)

    scipy.optimize.minimize(
        aggregate, model.design, constraints=constraints, options={"disp": False}
    )

    final_performance = model.evaluate()
    val = aggregate(model.design)
    print(f"Optimal design input: {model.design}")
    print(f"Params: {final_performance}\nValue: {val}")

    if plot:
        plots.plot_optimization_results(
            preferences, initial_performance, final_performance
        )


def get_constraints(problem, preferences):
    """
    Extract constraints given a set of class functions.

    Scipy needs a sequence of dicts describing inequalities that
    will be forced to be >=0.
    """
    print("Building constraints")
    constraints = []
    for funcname, func in preferences.items():
        if isinstance(func, classfunctions.MustBeAbove):
            # param >= cutoff implies param-cutoff>=0
            def constraint(x, prob, funcname):
                """Evaluate a greater-than constraint for optimizer."""
                prob.design = x
                return getattr(prob, funcname)() - func.bounds.cutoff  # pylint: disable=cell-var-from-loop
        elif isinstance(func, classfunctions.MustBeBelow):
            # param <= cutoff implies cutoff - param>=0
            def constraint(x, prob, funcname):
                """Evaluate a less-than constraint for optimizer."""
                prob.design = x
                return func.bounds.cutoff - getattr(prob, funcname)()  # pylint: disable=cell-var-from-loop
        else:
            constraint = None

        if constraint is not None:
            constraints.append(
                {"type": "ineq", "fun": constraint, "args": (problem, funcname)}
            )
    return constraints
