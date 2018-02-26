"""Optimize a design according to preferences."""
import scipy.optimize

from physprog import classfunctions
from physprog import objective
from physprog import plots


def optimize(model, preferences, plot=False):
    """Optimize the given problem to specified preferences."""
    classfunctions.build_all_splines(preferences.values())
    constraints = get_constraints(model, preferences)
    aggregate = objective.build_objective(model, preferences)
    initial_performance = model.evaluate()
    print('Optimizing design starting at value: {:.2f}\n{}'
          ''.format(aggregate(model.design), initial_performance))

    scipy.optimize.minimize(
        aggregate,
        model.design,
        constraints=constraints,
        options={'disp': False})

    final_performance = model.evaluate()
    print('Optimal design input: {}\nParams: {}\nValue: {}'
          ''.format(model.design, final_performance, aggregate(
              model.design)))

    if plot:
        plots.plot_optimization_results(
            preferences,
            initial_performance,
            final_performance)


def get_constraints(problem, preferences):
    """
    Extract constraints given a set of class functions.

    Scipy needs a sequence of dicts describing inequalities that
    will be forced to be >=0.
    """
    print('Building constraints')
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
            constraints.append({
                'type': 'ineq',
                'fun': constraint,
                'args': (problem, funcname)
            })
    return constraints
