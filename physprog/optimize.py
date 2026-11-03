"""Optimize a design according to preferences."""

import warnings

import scipy.optimize

from physprog import classfunctions, objective


def optimize(model, preferences, plot=False):
    """
    Optimize the given problem to specified preferences.

    The model is left at the optimal design and the
    :class:`scipy.optimize.OptimizeResult` is returned.
    """
    constraints = get_constraints(model, preferences)
    aggregate = objective.build_objective(model, preferences)
    initial_performance = _performance(model, preferences)
    val = aggregate(model.design)
    print(f"Optimizing design starting at value: {val:.2f}")
    print(initial_performance)

    result = scipy.optimize.minimize(
        aggregate, model.design, constraints=constraints, options={"disp": False}
    )
    if not result.success:
        warnings.warn(f"Optimization did not converge: {result.message}", stacklevel=2)

    # The constraint and objective callbacks mutate the model, so make sure it
    # is left at the optimum rather than wherever the optimizer last probed.
    model.evaluate(result.x)
    final_performance = _performance(model, preferences)
    print(f"Optimal design input: {model.design}")
    print(f"Params: {final_performance}\nValue: {result.fun}")

    if plot:
        from physprog import plots  # noqa: PLC0415  (lazy: keep matplotlib optional)

        plots.plot_optimization_results(
            preferences, initial_performance, final_performance
        )

    return result


def _performance(model, preferences):
    """Return the current value of each dependent variable, keyed by name."""
    return {name: getattr(model, name)() for name in preferences}


def get_constraints(problem, preferences):
    """
    Extract constraints given a set of class functions.

    Scipy needs a sequence of dicts describing inequalities that
    will be forced to be >=0.
    """
    print("Building constraints")
    constraints = []
    for funcname, func in preferences.items():
        # (cutoff, sign) pairs: param >= cutoff is sign +1 (param - cutoff >= 0)
        # and param <= cutoff is sign -1 (cutoff - param >= 0).
        if isinstance(func, classfunctions.MustBeAbove):
            limits = [(func.bounds.cutoff, 1.0)]
        elif isinstance(func, classfunctions.MustBeBelow):
            limits = [(func.bounds.cutoff, -1.0)]
        elif isinstance(func, classfunctions.MustBeInRange):
            limits = [(func.bounds.lower, 1.0), (func.bounds.upper, -1.0)]
        else:
            continue
        constraints.extend(
            {
                "type": "ineq",
                "fun": _constraint,
                "args": (problem, funcname, cutoff, sign),
            }
            for cutoff, sign in limits
        )
    return constraints


def _constraint(x, prob, funcname, cutoff, sign):
    """Evaluate a hard constraint; must be >= 0 when satisfied."""
    prob.design = x
    return sign * (getattr(prob, funcname)() - cutoff)
