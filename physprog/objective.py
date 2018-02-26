"""
Code for aggregate building the objective function from class functions.

This is useful once class functions have been built and you
just need the aggregate objective function. This is
used by the optimize module to perform optimization
but may also be used for any external optimization needs.
"""

from physprog import classfunctions


def build_objective(model, preferences):
    """Build an objective function based on a model and preferences."""
    def objective(x):
        """Evaluate the aggregate-objective function."""
        model.evaluate(x)
        total = 0.0
        for funcname, func in preferences.items():
            if issubclass(func.__class__, classfunctions.SmoothClassFunction):
                try:
                    # e.g. cost of this design
                    param_val = getattr(model, funcname)()
                except ValueError:
                    # invalid, throw a big penalty.
                    return 1e3
                total += func.evaluate(param_val)  # transformed cost
        return total

    return objective
