"""
Code for aggregate building the objective function from class functions.

This is useful once class functions have been built and you
just need the aggregate objective function. This is
used by the optimize module to perform optimization
but may also be used for any external optimization needs.
"""

import collections

import numpy as np

from physprog import classfunctions

# Penalty returned by the objective when the model rejects a design.
INVALID_DESIGN_PENALTY = 1e3


class ModelEvaluator:
    """
    Evaluate a model once per design and cache its dependent values.

    Optimizers call the objective and every constraint at the same design
    points (including the finite-difference points used for gradients), so
    sharing one evaluator means each unique design runs the model once.
    Calling it returns ``{name: value}`` for every dependent in
    ``preferences``. A ``ValueError`` from the model is cached and re-raised
    for that design.

    Parameters
    ----------
    model : object
        Has ``evaluate(x)``, which computes outputs for design ``x``, and a
        method per dependent variable returning its value.
    preferences : dict
        Class functions keyed by dependent-variable name.
    maxsize : int
        Number of recent designs to remember.
    """

    def __init__(self, model, preferences, maxsize=128):
        """Construct an evaluator."""
        self.model = model
        self.names = list(preferences)
        self.maxsize = maxsize
        self.model_evaluations = 0
        self._cache = collections.OrderedDict()

    def __call__(self, x):
        """Return the dependent values for design ``x``."""
        key = tuple(np.asarray(x, dtype=float).ravel())
        if key in self._cache:
            self._cache.move_to_end(key)
            result = self._cache[key]
        else:
            result = self._evaluate(x)
            self._cache[key] = result
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        if isinstance(result, ValueError):
            raise result
        return result

    def _evaluate(self, x):
        self.model_evaluations += 1
        try:
            self.model.evaluate(x)
            return {name: getattr(self.model, name)() for name in self.names}
        except ValueError as error:
            return error


def build_objective(model, preferences, evaluator=None):
    """
    Build an objective function based on a model and preferences.

    Pass an existing :class:`ModelEvaluator` to share its cache with the
    constraints from :func:`physprog.optimize.get_constraints`.
    """
    if evaluator is None:
        evaluator = ModelEvaluator(model, preferences)
    soft = {
        name: func
        for name, func in preferences.items()
        if isinstance(func, classfunctions.SmoothClassFunction)
    }

    def objective(x):
        """Evaluate the aggregate-objective function."""
        try:
            values = evaluator(x)
        except ValueError:
            # invalid, throw a big penalty.
            return INVALID_DESIGN_PENALTY
        # sum of transformed values, e.g. cost of this design
        return sum(func.evaluate(values[name]) for name, func in soft.items())

    return objective
