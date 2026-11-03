"""Regenerate the figures in ``assets/`` that are shown in the README."""

import pathlib

import matplotlib

matplotlib.use("Agg")

from physprog import classfunctions, optimize, plots
from physprog.tests.test_sample_problem import (
    SAMPLE_INPUT,
    SampleProblemBeam,
)

ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"


def main():
    """Build every README figure."""
    prefs = classfunctions.from_input(SAMPLE_INPUT)

    plots.plot_class_function(prefs["cost"], fname=ASSETS / "class-function-cost.png")
    plots.plot_class_function(
        prefs["frequency"], fname=ASSETS / "class-function-frequency.png"
    )

    beam = SampleProblemBeam()
    initial = {name: getattr(beam, name)() for name in prefs}
    optimize.optimize(beam, prefs)
    final = {name: getattr(beam, name)() for name in prefs}

    plots.plot_class_functions(
        prefs, initial, final, fname=ASSETS / "sample-class-functions.png"
    )
    plots.plot_optimization_results(
        prefs, initial, final, fname=ASSETS / "sample-results.png"
    )


if __name__ == "__main__":
    main()
