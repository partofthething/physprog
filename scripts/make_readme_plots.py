"""Regenerate the figures in ``assets/`` that are shown in the README."""

import pathlib

import matplotlib
import matplotlib.pyplot as plt

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

    plot_two_sided()

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


def plot_two_sided():
    """Show value-is-better and range-is-better side by side."""
    value = classfunctions.ValueIsBetter(
        [4.6, 4.2, 4.0, 3.8, 3.5], [5.2, 5.6, 5.8, 6.0, 6.3], target=4.9
    )
    value.name, value.units = "thickness", "mm"
    in_range = classfunctions.RangeIsBetter([20, 18, 17, 16, 15], [24, 26, 27, 28, 29])
    in_range.name, in_range.units = "temperature", "°C"
    classfunctions.build_all_splines([value, in_range])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plots.plot_class_function(value, ax=axes[0])
    plots.plot_class_function(in_range, ax=axes[1])
    fig.tight_layout()
    fig.savefig(ASSETS / "class-function-two-sided.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
