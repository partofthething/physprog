"""Optimize a design according to preferences."""

import warnings

import numpy as np
import scipy.optimize

from physprog import classfunctions, objective


def optimize(model, preferences, plot=False):
    """
    Optimize the given problem to specified preferences.

    The model is left at the optimal design and the
    :class:`scipy.optimize.OptimizeResult` is returned.
    """
    evaluator = objective.ModelEvaluator(model, preferences)
    constraints = get_constraints(model, preferences, evaluator=evaluator)
    aggregate = objective.build_objective(model, preferences, evaluator=evaluator)
    initial_performance = dict(evaluator(model.design))
    val = aggregate(model.design)
    print(f"Optimizing design starting at value: {val:.2f}")
    print(initial_performance)

    result = scipy.optimize.minimize(
        aggregate, model.design, constraints=constraints, options={"disp": False}
    )
    if not result.success:
        warnings.warn(f"Optimization did not converge: {result.message}", stacklevel=2)

    # The callbacks leave the model at whichever design was evaluated last,
    # so make sure it ends at the optimum.
    model.evaluate(result.x)
    final_performance = _performance(model, preferences)
    print(f"Model evaluations: {evaluator.model_evaluations + 1}")
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


def get_constraints(problem, preferences, evaluator=None):
    """
    Extract constraints given a set of class functions.

    Returns a list for ``scipy.optimize.minimize(constraints=...)`` with at
    most one vector-valued "ineq" constraint (each entry >= 0 when
    satisfied) and one "eq" constraint (each entry == 0).

    Pass the result as ``constraints=`` to ``scipy.optimize.minimize`` when
    using the aggregate objective from :mod:`physprog.objective` directly;
    hard preferences are not part of the aggregate objective. Passing the
    same :class:`~physprog.objective.ModelEvaluator` to both lets them share
    model evaluations.
    """
    print("Building constraints")
    if evaluator is None:
        evaluator = objective.ModelEvaluator(problem, preferences)
    # (name, cutoff, sign): param >= cutoff is sign +1 (param - cutoff >= 0)
    # and param <= cutoff is sign -1 (cutoff - param >= 0).
    limits = {"ineq": [], "eq": []}
    for name, func in preferences.items():
        if isinstance(func, classfunctions.MustBeAbove):
            limits["ineq"].append((name, func.bounds.cutoff, 1.0))
        elif isinstance(func, classfunctions.MustBeBelow):
            limits["ineq"].append((name, func.bounds.cutoff, -1.0))
        elif isinstance(func, classfunctions.MustEqual):
            limits["eq"].append((name, func.bounds.cutoff, 1.0))
        elif isinstance(func, classfunctions.MustBeInRange):
            limits["ineq"].append((name, func.bounds.lower, 1.0))
            limits["ineq"].append((name, func.bounds.upper, -1.0))
    return [
        {"type": kind, "fun": _constraints, "args": (evaluator, entries)}
        for kind, entries in limits.items()
        if entries
    ]


def _constraints(x, evaluator, entries):
    """Evaluate hard constraints; each >= 0 (or == 0 for "eq") when satisfied."""
    values = evaluator(x)
    return np.array([sign * (values[name] - cutoff) for name, cutoff, sign in entries])
