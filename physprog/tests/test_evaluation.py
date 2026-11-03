"""Tests for how often and how correctly the model gets evaluated."""

import pytest

from physprog import classfunctions as cf
from physprog import objective, optimize


class StatefulModel:
    """
    Model that only computes outputs inside ``evaluate``, like a simulation.

    Output methods return stored results, so anything that reads them
    without calling ``evaluate(x)`` first gets values for an old design.
    """

    def __init__(self):
        self.design = [3.0, 3.0]
        self.runs = 0
        self._results = {}
        self.evaluate()

    def evaluate(self, x=None):
        if x is not None:
            self.design = list(x)
        self.runs += 1
        a, b = self.design
        self._results = {"area": a * b, "perimeter": 2 * (a + b), "a": a}

    def area(self):
        return self._results["area"]

    def perimeter(self):
        return self._results["perimeter"]

    def a(self):
        return self._results["a"]


def preferences():
    prefs = {
        "area": cf.LargerBetter(cf.SoftBounds(30, 25, 20, 15, 10)),
        "perimeter": cf.MustBeBelow(cf.HardBounds(20.0)),
        "a": cf.MustEqual(cf.HardBounds(4.0)),
    }
    cf.build_all_splines(prefs.values())
    return prefs


def test_constraints_see_the_design_being_asked_about():
    """Constraints must evaluate x, not read results left from another x."""
    model = StatefulModel()
    ineq, eq = optimize.get_constraints(model, preferences())
    model.evaluate([1.0, 1.0])  # leave the model somewhere else
    assert ineq["fun"]([4.0, 5.0], *ineq["args"]) == pytest.approx([20.0 - 18.0])
    assert eq["fun"]([4.0, 5.0], *eq["args"]) == pytest.approx([0.0])


def test_optimum_respects_hard_constraints_on_stateful_model():
    model = StatefulModel()
    result = optimize.optimize(model, preferences())
    assert result.success
    # a == 4 and perimeter <= 20 means b <= 6, and larger area is better
    assert model.design == pytest.approx([4.0, 6.0], abs=1e-4)
    assert model.perimeter() <= 20.0 + 1e-6


def test_each_design_runs_the_model_once():
    model = StatefulModel()
    prefs = preferences()
    evaluator = objective.ModelEvaluator(model, prefs)
    aggregate = objective.build_objective(model, prefs, evaluator=evaluator)
    constraints = optimize.get_constraints(model, prefs, evaluator=evaluator)
    x = [4.0, 5.0]
    aggregate(x)
    for constraint in constraints:
        constraint["fun"](x, *constraint["args"])
    assert evaluator.model_evaluations == 1


def test_optimize_shares_evaluations():
    """Model runs stay close to the number of distinct designs tried."""
    seen = set()

    class Tracking(StatefulModel):
        def evaluate(self, x=None):
            super().evaluate(x)
            seen.add(tuple(self.design))

    model = Tracking()
    optimize.optimize(model, preferences())
    # one run per distinct design, plus the constructor and final re-evaluation
    assert model.runs <= len(seen) + 2


def test_invalid_design_penalized_and_cached():
    class Fragile(StatefulModel):
        def evaluate(self, x=None):
            super().evaluate(x)
            if self.design[0] < 0:
                raise ValueError("negative side")

    model = Fragile()
    prefs = preferences()
    evaluator = objective.ModelEvaluator(model, prefs)
    aggregate = objective.build_objective(model, prefs, evaluator=evaluator)
    assert aggregate([-1.0, 2.0]) == objective.INVALID_DESIGN_PENALTY
    assert aggregate([-1.0, 2.0]) == objective.INVALID_DESIGN_PENALTY
    assert evaluator.model_evaluations == 1
    with pytest.raises(ValueError):
        evaluator([-1.0, 2.0])
