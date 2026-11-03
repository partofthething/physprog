"""Tests for two-sided (value/range is better, must be in range) preferences."""

import numpy as np
import pytest
import scipy.optimize

from physprog import classfunctions as cf
from physprog import objective, optimize

LOWER = (4.5, 4.0, 3.5, 3.0, 2.5)
UPPER = (5.5, 6.0, 6.5, 7.0, 7.5)


def build(func, nsc_copies=3):
    cf.build_all_splines([func] * nsc_copies)
    return func


@pytest.fixture(
    params=[
        lambda: cf.ValueIsBetter(LOWER, UPPER),
        lambda: cf.ValueIsBetter(LOWER, UPPER, target=4.7),
        lambda: cf.ValueIsBetter(LOWER, UPPER, pull=0.0),
        lambda: cf.ValueIsBetter(LOWER, UPPER, pull=0.95),
        lambda: cf.ValueIsBetter((4.5, 4.4, 4.3, 4.2, 4.1), UPPER),
        lambda: cf.RangeIsBetter(LOWER, UPPER),
    ],
    ids=[
        "value",
        "value-off-center",
        "value-flat",
        "value-deep",
        "value-lopsided",
        "range",
    ],
)
def func(request):
    return build(request.param())


def best_point(func):
    if isinstance(func, cf.ValueIsBetter):
        return func.target
    return sum(func.best_region) / 2


def test_which_region():
    func = cf.RangeIsBetter(LOWER, UPPER)
    assert func.which_region(5.0) == cf.AWESOME
    assert func.which_region(4.2) == cf.DESIRABLE
    assert func.which_region(6.2) == cf.TOLERABLE
    assert func.which_region(2.0) == cf.UNACCEPTABLE
    assert func.which_region(8.0) == cf.UNACCEPTABLE


def test_both_sides_share_values(func):
    np.testing.assert_allclose(func.lower.gis, func.upper.gis)


def one_sided_slopes(func, x, h=1e-6):
    """Second-order accurate slopes just left and right of x."""
    f = func.evaluate
    left = (3 * f(x) - 4 * f(x - h) + f(x - 2 * h)) / (2 * h)
    right = (-3 * f(x) + 4 * f(x + h) - f(x + 2 * h)) / (2 * h)
    return left, right


def test_continuous_and_smooth(func):
    eps = 1e-9
    for x in [*func.bounds, best_point(func)]:
        assert func.evaluate(x - eps) == pytest.approx(func.evaluate(x + eps), rel=1e-6)
        left, right = one_sided_slopes(func, x)
        assert left == pytest.approx(right, rel=1e-3, abs=1e-6)


def test_gets_worse_away_from_best(func):
    """Monotonic on each side of the best point, positive everywhere."""
    best = best_point(func)
    left = np.linspace(func.bounds[0] - 2, best, 1000)
    right = np.linspace(best, func.bounds[-1] + 2, 1000)
    left_vals = [func.evaluate(g) for g in left]
    right_vals = [func.evaluate(g) for g in right]
    assert min(left_vals + right_vals) > 0
    assert np.all(np.diff(left_vals) <= 1e-12)
    assert np.all(np.diff(right_vals) >= -1e-12)


def test_value_has_minimum_at_target():
    func = build(cf.ValueIsBetter(LOWER, UPPER, target=4.7))
    assert func.evaluate(4.7) < func.evaluate(4.6)
    assert func.evaluate(4.7) < func.evaluate(4.8)
    assert func.acceptability(4.7) == pytest.approx(1.0)


def test_default_target_is_midpoint_for_symmetric_bounds():
    func = build(cf.ValueIsBetter(LOWER, UPPER))
    assert func.target == pytest.approx(5.0)


def test_pull_sets_depth():
    shallow = build(cf.ValueIsBetter(LOWER, UPPER, pull=0.1))
    deep = build(cf.ValueIsBetter(LOWER, UPPER, pull=0.8))
    g0 = deep.gis[0]
    assert deep.evaluate(5.0) == pytest.approx(0.2 * g0)
    assert shallow.evaluate(5.0) == pytest.approx(0.9 * g0)


class Competing:
    """Thickness has a target, but cost (100 x thickness) wants it thinner."""

    def __init__(self):
        self.design = np.array([5.15])

    def evaluate(self, x=None):
        if x is not None:
            self.design = np.asarray(x, dtype=float)

    def thickness(self):
        return self.design[0]

    def cost(self):
        return 100 * self.design[0]


@pytest.mark.parametrize(("pull", "near_target"), [(0.0, False), (0.5, True)])
def test_pull_resists_competing_metric(pull, near_target):
    lower, upper = (4.6, 4.2, 4.0, 3.8, 3.5), (5.2, 5.6, 5.8, 6.0, 6.3)
    prefs = {
        "thickness": cf.ValueIsBetter(lower, upper, target=4.9, pull=pull),
        "cost": cf.SmallerBetter(cf.SoftBounds(300, 600, 650, 700, 750)),
    }
    cf.build_all_splines(prefs.values())
    model = Competing()
    optimize.optimize(model, prefs)
    assert bool(abs(model.thickness() - 4.9) < 0.15) == near_target


def test_range_is_flat_inside():
    func = build(cf.RangeIsBetter(LOWER, UPPER))
    values = {func.evaluate(g) for g in np.linspace(4.5, 5.5, 11)}
    assert values == {func.gis[0]}


def test_acceptability(func):
    xs = np.linspace(func.bounds[0] - 1, func.bounds[-1] + 1, 500)
    values = [func.acceptability(g) for g in xs]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert values[0] == values[-1] == 0.0


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((LOWER, (4.0, 6.0, 6.5, 7.0, 7.5)), "below the first upper bound"),
        ((LOWER, UPPER, 6.0), "strictly between"),
        ((LOWER, UPPER, None, 1.0), "pull"),
        (((4.5, 4.0, 3.5, 3.0, 3.1), UPPER), "strictly"),
    ],
)
def test_bad_inputs_rejected(args, match):
    with pytest.raises(ValueError, match=match):
        cf.ValueIsBetter(*args)


def test_must_be_in_range():
    func = cf.MustBeInRange(cf.RangeBounds(1.0, 2.0))
    assert func.acceptability(1.5) == 1.0
    assert func.acceptability(2.5) == 0.0
    with pytest.raises(ValueError):
        cf.MustBeInRange(cf.RangeBounds(2.0, 1.0))


def test_must_equal_acceptability():
    func = cf.MustEqual(cf.HardBounds(4.0))
    assert func.acceptability(4.0) == 1.0
    assert func.acceptability(4.0 + 1e-12) == 1.0
    assert func.acceptability(4.1) == 0.0


def test_must_equal_is_an_equality_constraint():
    """
    The hard equality on angle (thickness + temperature) wins.

    Both soft preferences can't stay highly desirable (5.5 + 24 < 30), so
    the optimizer trades them off but neither falls past desirable.
    """
    prefs = {
        "thickness": cf.ValueIsBetter(LOWER, UPPER),
        "temperature": cf.RangeIsBetter((20, 18, 16, 14, 12), (24, 26, 28, 30, 32)),
        "angle": cf.MustEqual(cf.HardBounds(30.0)),
    }
    cf.build_all_splines(prefs.values())
    model = Model()
    constraints = optimize.get_constraints(model, prefs)
    assert [c["type"] for c in constraints] == ["eq"]
    result = optimize.optimize(model, prefs)
    assert result.success
    assert model.angle() == pytest.approx(30.0, abs=1e-6)
    for name in ("thickness", "temperature"):
        region = prefs[name].which_region(getattr(model, name)())
        assert region <= cf.DESIRABLE, name


def test_from_input(tmp_path):
    inp = tmp_path / "two-sided.yaml"
    inp.write_text(
        """
dependents:
  - name: thickness
    class: ValueIsBetter
    units: mm
    target: 4.9
    pull: 0.7
    lower_bounds: [4.5, 4.0, 3.5, 3.0, 2.5]
    upper_bounds: [5.5, 6.0, 6.5, 7.0, 7.5]
  - name: temperature
    class: RangeIsBetter
    lower_bounds: [20, 18, 16, 14, 12]
    upper_bounds: [24, 26, 28, 30, 32]
  - name: angle
    class: MustBeInRange
    bounds: [10, 20]
  - name: span
    class: MustEqual
    bound: 4.0
"""
    )
    funcs = cf.from_input(inp)
    assert funcs["thickness"].target == 4.9
    assert funcs["thickness"].pull == 0.7
    assert funcs["thickness"].units == "mm"
    assert funcs["temperature"].best_region == (20, 24)
    assert funcs["angle"].bounds == cf.RangeBounds(10, 20)
    assert isinstance(funcs["span"], cf.MustEqual)
    assert funcs["span"].bounds.cutoff == 4.0


class Model:
    """Toy model whose dependents are simple functions of two inputs."""

    def __init__(self):
        self.design = np.array([1.0, 9.0])

    def evaluate(self, x=None):
        if x is not None:
            self.design = np.asarray(x, dtype=float)
        return [self.thickness(), self.temperature(), self.angle()]

    def thickness(self):
        return self.design[0]

    def temperature(self):
        return self.design[1]

    def angle(self):
        return self.design[0] + self.design[1]


def test_optimizes_to_target_and_range():
    prefs = {
        "thickness": cf.ValueIsBetter(LOWER, UPPER, target=4.8),
        "temperature": cf.RangeIsBetter((20, 18, 16, 14, 12), (24, 26, 28, 30, 32)),
        "angle": cf.MustBeInRange(cf.RangeBounds(10.0, 27.0)),
    }
    cf.build_all_splines(prefs.values())
    model = Model()
    result = optimize.optimize(model, prefs)
    assert result.success
    # The best region is deliberately shallow, so the optimizer only needs to
    # get thickness into it, not exactly onto the target.
    assert prefs["thickness"].which_region(model.thickness()) == cf.AWESOME
    assert 20 <= model.temperature() <= 24
    assert 10 - 1e-6 <= model.angle() <= 27 + 1e-6


def test_range_constraint_binds():
    """With the range constraint tight, temperature is pulled out of its range."""
    prefs = {
        "thickness": cf.ValueIsBetter(LOWER, UPPER),
        "temperature": cf.RangeIsBetter((20, 18, 16, 14, 12), (24, 26, 28, 30, 32)),
        "angle": cf.MustBeInRange(cf.RangeBounds(10.0, 22.0)),
    }
    cf.build_all_splines(prefs.values())
    model = Model()
    (constraint,) = optimize.get_constraints(model, prefs)
    # lower and upper limits of the range
    assert len(constraint["fun"](model.design, *constraint["args"])) == 2
    constraints = [constraint]
    aggregate = objective.build_objective(model, prefs)
    result = scipy.optimize.minimize(aggregate, model.design, constraints=constraints)
    model.evaluate(result.x)
    assert model.angle() == pytest.approx(22.0, abs=1e-4)
