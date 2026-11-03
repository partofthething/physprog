"""Run a sample problem to test full system."""

import math
import os
import unittest
from collections import namedtuple

import numpy as np
import pytest

from physprog import classfunctions, optimize

THIS_DIR = os.path.dirname(__file__)

SAMPLE_INPUT = os.path.join(THIS_DIR, "sample-input.yaml")


class TestInput(unittest.TestCase):
    """Test that input can be read."""

    def test_read_class_functions(self):
        functions = classfunctions.from_input(SAMPLE_INPUT)
        self.assertTrue("frequency" in functions)


class Test_Sample_Problem(unittest.TestCase):
    """Test by optimizing a beam problem from the literature."""

    def test_optimization(self):
        beam = SampleProblemBeam()
        # check initial conditions
        self.assertAlmostEqual(beam.frequency(), 113.0, delta=0.5)
        self.assertAlmostEqual(beam.cost(), 1060.0)
        self.assertAlmostEqual(beam.mass(), 2230.0)

        prefs = classfunctions.from_input(SAMPLE_INPUT)
        result = optimize.optimize(beam, prefs, plot=False)

        self.assertTrue(result.success)
        # model is left at the optimum the optimizer reported
        np.testing.assert_allclose(beam.design, result.x)
        # not rigorous, but happens in this problem
        self.assertLess(beam.cost(), 1060.0)
        # every soft preference ends up at least tolerable
        for name, func in prefs.items():
            if isinstance(func, classfunctions.SmoothClassFunction):
                region = func.which_region(getattr(beam, name)())
                self.assertLessEqual(region, classfunctions.TOLERABLE, name)


def test_constraints_use_their_own_cutoffs():
    """Each hard constraint must use its own cutoff (no late binding)."""
    beam = SampleProblemBeam()  # width=0.4, length=5.0
    prefs = {
        "width": classfunctions.MustBeAbove(classfunctions.HardBounds(0.1)),
        "length": classfunctions.MustBeBelow(classfunctions.HardBounds(7.0)),
    }
    constraints = optimize.get_constraints(beam, prefs)
    values = [c["fun"](beam.design, *c["args"]) for c in constraints]
    assert values == pytest.approx([0.4 - 0.1, 7.0 - 5.0])


def test_plots_render(tmp_path):
    """Smoke test the plotting helpers on the sample problem."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    from physprog import plots  # noqa: PLC0415

    prefs = classfunctions.from_input(SAMPLE_INPUT)
    beam = SampleProblemBeam()
    initial = {name: getattr(beam, name)() for name in prefs}
    beam.design = (0.3, 0.32, 0.36, 0.6, 3.0)  # length unacceptable
    final = {name: getattr(beam, name)() for name in prefs}

    plots.plot_class_function(prefs["cost"], fname=tmp_path / "one.png")
    plots.plot_class_functions(prefs, initial, final, fname=tmp_path / "grid.png")
    plots.plot_optimization_results(prefs, initial, final, fname=tmp_path / "r.png")
    assert len(list(tmp_path.glob("*.png"))) == 3


SampleDesign = namedtuple("SampleDesign", ["d1", "d2", "d3", "b", "L"])


class SampleProblemBeam:
    """Sample beam design problem from Messac, 1996."""

    E1 = 1.6e9
    C1 = 500.0
    RHO1 = 100.0
    E2 = 70e9
    C2 = 1500.0
    RHO2 = 2770.0
    E3 = 200e9
    C3 = 800.0
    RHO3 = 7780.0

    def __init__(self):
        self._design = SampleDesign(0.3, 0.35, 0.40, 0.40, 5.0)  # initial

    def evaluate(self, x=None):
        """Convert input design into output design parameters."""
        if x is not None:
            self.design = x
        return [
            self.frequency(),
            self.cost(),
            self.width(),
            self.length(),
            self.mass(),
            self.semiheight(),
            self.width_layer1(),
            self.width_layer2(),
            self.width_layer3(),
        ]

    @property
    def design(self):
        return self._design

    @design.setter
    def design(self, val):
        self._design = SampleDesign(*val)

    @property
    def ei(self):
        ds = self.design
        return (
            2.0
            / 3.0
            * ds.b
            * (
                self.E1 * ds.d1**3
                + self.E2 * (ds.d2**3 - ds.d1**3)
                + self.E3 * (ds.d3**3 - ds.d2**3)
            )
        )

    @property
    def mu(self):
        ds = self.design
        return (
            2
            * ds.b
            * (
                self.RHO1 * ds.d1
                + self.RHO2 * (ds.d2 - ds.d1)
                + self.RHO3 * (ds.d3 - ds.d2)
            )
        )

    def frequency(self):
        return math.pi / (2 * self.design.L**2) * math.sqrt(self.ei / self.mu)

    def cost(self):
        ds = self.design
        # cost in the paper says 1060 but I'm getting 212, exactly a
        # factor of 5 off. But why?? Ah, because cost should have L in it!
        # That's a typo in the paper.
        return (
            2
            * ds.b
            * ds.L
            * (self.C1 * ds.d1 + self.C2 * (ds.d2 - ds.d1) + self.C3 * (ds.d3 - ds.d2))
        )

    def width(self):
        return self.design.b

    def length(self):
        return self.design.L

    def mass(self):
        return self.mu * self.design.L

    def semiheight(self):
        return self.design.d3

    def width_layer1(self):
        return self.design.d1

    def width_layer2(self):
        return self.design.d2 - self.design.d1

    def width_layer3(self):
        return self.design.d3 - self.design.d2


if __name__ == "__main__":
    unittest.main()
