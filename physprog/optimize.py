"""Run optimization problems."""
import scipy.optimize

from physprog import classfunctions
from physprog import plots


def optimize(problem, preferences):
    """Optimize the given problem to specified preferences."""
    classfunctions.construct_splines(preferences.values())
    constraints = get_constraints(problem, preferences)

    def objective(x):
        """Evaluate the aggregate-objective function."""
        problem.design = x
        total = 0.0
        for funcname, func in preferences.items():
            if issubclass(func.__class__, classfunctions.SmoothClassFunction):
                try:
                    param_val = getattr(problem,
                                        funcname)()  # e.g. cost of this design
                except ValueError:
                    # invalid, throw a big penalty.
                    return 1e3

                total += func.evaluate(param_val)  # transformed cost
        return total

    initial_performance = problem.analyze()
    print('Optimizing design starting at value: {:.2f}\n{}'
          ''.format(objective(problem.design), initial_performance))
    scipy.optimize.minimize(
        objective,
        problem.design,
        constraints=constraints,
        options={'disp': True})

    final_performance = problem.analyze()
    print('Optimal design input: {}\nParams: {}\nValue: {}'
          ''.format(problem.design, final_performance, objective(
              problem.design)))

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
