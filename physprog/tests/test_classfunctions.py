"""Unit tests for class functions."""

import numpy as np
import pytest

from physprog import classfunctions as cf

SMALLER = (cf.SmallerBetter, (10, 20, 30, 40, 50))
LARGER = (cf.LargerBetter, (50, 40, 30, 20, 10))


def build(cls, bounds, nsc_copies=3):
    """Build a class function as if it were one of several soft functions."""
    func = cls(cf.SoftBounds(*bounds))
    cf.build_all_splines([func] * nsc_copies)
    return func


@pytest.fixture(params=[SMALLER, LARGER], ids=["smaller", "larger"])
def func(request):
    cls, bounds = request.param
    return build(cls, bounds)


def worst_ward(func, pad=20.0, n=2000):
    """Sample points from far on the best side to far on the worst side."""
    sign = np.sign(func.bounds[-1] - func.bounds[0])
    return np.linspace(func.bounds[0] - sign * pad, func.bounds[-1] + sign * pad, n)


def test_which_region_smaller():
    sb = cf.SmallerBetter(cf.SoftBounds(10, 20, 30, 40, 50))
    assert sb.which_region(0) == cf.AWESOME
    assert sb.which_region(35) == cf.UNDESIRABLE
    assert sb.which_region(55) == cf.UNACCEPTABLE


def test_which_region_larger():
    lb = cf.LargerBetter(cf.SoftBounds(50, 40, 30, 20, 10))
    assert lb.which_region(55) == cf.AWESOME
    assert lb.which_region(35) == cf.TOLERABLE
    assert lb.which_region(5) == cf.UNACCEPTABLE


def test_values_at_bounds_match_targets(func):
    for bound, target in zip(func.bounds, func.gis, strict=True):
        assert func.evaluate(bound) == pytest.approx(target)


def test_continuous_and_smooth_at_bounds(func):
    """Value and slope match on each side of every bound (C1 continuity)."""
    eps = 1e-6
    h = 1e-4
    for bound in func.bounds:
        left, right = func.evaluate(bound - eps), func.evaluate(bound + eps)
        assert left == pytest.approx(right, rel=1e-4)
        dleft = (func.evaluate(bound - eps) - func.evaluate(bound - eps - h)) / h
        dright = (func.evaluate(bound + eps + h) - func.evaluate(bound + eps)) / h
        assert dleft == pytest.approx(dright, rel=1e-2, abs=1e-6)


def test_positive_and_monotonic(func):
    """Values are positive and get strictly worse moving to the worst side."""
    values = np.array([func.evaluate(g) for g in worst_ward(func)])
    assert np.all(values > 0)
    assert np.all(np.diff(values) > 0)


def test_unacceptable_is_heavily_penalized(func):
    sign = np.sign(func.bounds[-1] - func.bounds[0])
    just_past = func.evaluate(func.bounds[-1] + sign * 1.0)
    far_past = func.evaluate(func.bounds[-1] + sign * 20.0)
    assert just_past > func.gis[-1]
    assert far_past > 10 * just_past


def test_acceptability_in_unit_interval(func):
    values = [func.acceptability(g) for g in worst_ward(func)]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert np.all(np.diff(values) <= 1e-12)
    sign = np.sign(func.bounds[-1] - func.bounds[0])
    assert func.acceptability(func.bounds[-1] + sign) == 0.0


def test_from_input_rejects_unknown_class(tmp_path):
    inp = tmp_path / "bad.yaml"
    inp.write_text("dependents:\n  - name: x\n    class: np\n    bound: 1\n")
    with pytest.raises(ValueError, match="Unknown class function"):
        cf.from_input(inp)


@pytest.mark.parametrize(
    ("cls", "bounds"),
    [
        (cf.SmallerBetter, (10, 20, 30, 50, 40)),
        (cf.LargerBetter, (10, 20, 30, 40, 50)),
        (cf.SmallerBetter, (10, 20, 30, 40)),
    ],
)
def test_misordered_bounds_rejected(cls, bounds):
    with pytest.raises(ValueError, match="strictly"):
        cls(bounds)
