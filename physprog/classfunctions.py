"""Definitions of the various kinds of Class Functions."""

import collections
import math
import sys

import numpy as np
import yaml

# data structure representing the limits of each range.
# right-hand limit is either positive or negative infinity.
# we order them consistently with the region numbering, so
# the most desirable regions are first.
SoftBounds = collections.namedtuple(
    "SoftBounds", ["awesome", "desirable", "tolerable", "undesirable", "horrible"]
)
HardBounds = collections.namedtuple("HardBounds", ["cutoff"])

# region labels
AWESOME = 0
DESIRABLE = 1
TOLERABLE = 2
UNDESIRABLE = 3
HORRIBLE = 4
UNACCEPTABLE = 5

MAX_ITERATIONS = 20

# Largest beta tried when building splines (beta grows by 0.5 per attempt).
MAX_BETA = 40.0

# Quadratic coefficient of the penalty applied beyond the last bound. The
# penalty continues the last spline with matching value and slope, then adds
# this times the square of the overshoot (in class-function units), so it is
# smooth (C1) for gradient-based optimizers but grows quickly.
UNACCEPTABLE_PENALTY = 10.0

REGION_NAMES = (
    "Highly desirable",
    "Desirable",
    "Tolerable",
    "Undesirable",
    "Highly undesirable",
    "Unacceptable",
)


class ClassFunction:
    """Basic class function expressing preferences."""

    # Optional metadata used for labeling plots.
    name = None
    units = None

    def __init__(self, bounds):
        """Construct class function."""
        self.bounds = bounds

    def evaluate(self, g):
        """Evaluate the class function against some input."""
        raise NotImplementedError

    def build_splines(self, nsc, alpha=0.05, initial_beta=1.5):
        """Compute the splines that define this class function's values."""
        raise NotImplementedError


class SmoothClassFunction(ClassFunction):
    """
    Smooth constraint: higher/lower/value is better.

    This is made of splines that are computed in an algorithm.
    """

    def __init__(self, bounds):
        """Construct class function."""
        ClassFunction.__init__(self, bounds)
        self.slopes = []
        self.region_slopes = []
        self.gis = []
        self.dgis = []

    def evaluate(self, g):
        """
        Return the computed objective value of dependent value g.

        Uses the a,b,c,d coefficients in the full spline expression.
        """
        i = self.which_region(g)
        if i == AWESOME:
            # Exponential decay that matches the value and slope at the
            # first bound and never reaches zero.
            g0 = self.gis[0]
            return g0 * math.exp(self.slopes[0] / g0 * (g - self.bounds[0]))
        if i == UNACCEPTABLE:
            # Continue the last spline smoothly, then penalize quadratically.
            # slope * overshoot is positive for both smaller/larger-is-better.
            excess = self.slopes[-1] * (g - self.bounds[-1])
            return self.gis[-1] + excess + UNACCEPTABLE_PENALTY * excess**2
        a, b, c, d = self.evaluate_spline_coeffs(i)
        xi, width = self.get_region_fraction(g, i)
        xi4 = xi**4
        xim14 = (xi - 1) ** 4

        return width**4 * (a / 12.0 * xi4 + b / 12.0 * xim14) + c * width * xi + d

    def get_region_fraction(self, g, i):
        """Compute the fraction across region i the value g is."""
        if i == 0:
            return 0.5, 1.0  # undefined size
        bound_i = self.bounds[i]
        bound_im1 = self.bounds[i - 1]
        width = bound_i - bound_im1
        return (g - bound_im1) / width, width

    def build_splines(self, nsc, alpha=0.05, initial_beta=1.5):
        """Compute the splines that define this class function's values."""
        # start at 1.5, increase by 0.5 as recommended in paper
        betas = np.arange(initial_beta, MAX_BETA, 0.5)
        for beta in betas:
            self.gis = []
            self.dgis = []
            self.slopes = [0.0 for _bi in self.bounds]
            self.region_slopes = []
            self._compute_region_slopes(nsc, beta)
            acceptable = self._compute_pointwise_slopes(alpha)
            if acceptable:
                return beta

        raise RuntimeError("Class function construction did not converge.")

    def _compute_region_slopes(self, nsc, beta):
        """Perform first loop in spline-building iteration."""
        for i, _bpv in enumerate(self.bounds):
            if i == 0:
                gi = dgi = 0.1
                region_slope = 0.0
            else:
                dgi = beta * nsc * dgi
                gi = gi + dgi
                region_slope = dgi / (self.bounds[i] - self.bounds[i - 1])
            self.gis.append(gi)
            self.dgis.append(dgi)
            self.region_slopes.append(region_slope)

    def _compute_pointwise_slopes(self, alpha):
        """
        Perform second loop in computation of splines.

        We loop in a second loop because the first region's
        slope depends on the values from the 2nd region.
        """
        for i, region_slope in enumerate(self.region_slopes):
            if i == 0:
                # s_1 = alpha * (average slope of the next region), per Messac.
                self.slopes[i] = alpha * self.region_slopes[1]
            else:
                # slope[k]=skMin+alpha*(-skMax+skMin)
                # skMin-skMax in the paper. But if you calculate it you see
                # the paper is wrong, but the expression is right.
                slope_min_i = (4.0 * region_slope - self.slopes[i - 1]) / 3.0
                slope_max_i = 4.0 * region_slope - 3 * self.slopes[i - 1]
                self.slopes[i] = slope_min_i + alpha * (slope_max_i - slope_min_i)

                # check positivity of a and b constants.
                a, b, _c, _d = self.evaluate_spline_coeffs(i)
                if a < 0 or b < 0:
                    # no good, go to next beta.
                    break
        else:
            # if we didn't break in any region, this beta is appropriate.
            return True

        return False  # This beta is invalid

    def evaluate_spline_coeffs(self, i):
        """Compute the a,b,c,d, coefficients of the splines in all regions."""
        width = self.bounds[i] - self.bounds[i - 1]  # region size, lambda
        si = self.slopes[i]
        sim1 = self.slopes[i - 1]
        rsi = self.region_slopes[i]
        gim1 = self.gis[i - 1]

        a = (3.0 * (3.0 * si + sim1) - 12.0 * rsi) / (2.0 * width**3)
        b = (12.0 * rsi - 3.0 * (si + 3.0 * sim1)) / (2.0 * width**3)
        c = 2 * rsi - 0.5 * si - 0.5 * sim1
        d = gim1 - width * (0.5 * rsi - 0.125 * si - 0.375 * sim1)

        return a, b, c, d

    def which_region(self, g):
        """Determine which region the value g is in."""
        raise NotImplementedError

    def plot(self, fname=None):
        """Plot this class function, saving to ``fname`` or showing it."""
        from physprog import plots  # noqa: PLC0415  (lazy: keep matplotlib optional)

        plots.plot_class_function(self, fname=fname)

    def acceptability(self, g):
        """
        Rank the acceptability of value g on scale from 0 to 1.

        Useful for plotting relative changes to values.
        """
        region = self.which_region(g)
        if region == UNACCEPTABLE:
            return 0.0
        xi, _width = self.get_region_fraction(g, region)
        return 1.0 - (region + xi) / 5.0


class SmallerBetter(SmoothClassFunction):
    """
    Smaller values are better (1-S).

    Bounds input will have awesome as the lowest value, and others
    will be increasing.
    """

    def which_region(self, g):
        """Determine which region dependent variable value g is in."""
        low = -float("inf")
        for i, upper_edge in enumerate(self.bounds):
            if low <= g <= upper_edge:
                # this is our region.
                return i
            low = upper_edge

        return UNACCEPTABLE  # unacceptable.


class LargerBetter(SmoothClassFunction):
    """
    Larger values are better (2-S).

    Input bounds will have awesome as the highest value and other decreasing.
    """

    def which_region(self, g):
        """Determine which region dependent variable value g is in."""
        high = +float("inf")
        for i, lower_edge in enumerate(self.bounds):
            if lower_edge <= g <= high:
                # this is our region.
                return i
            high = lower_edge

        return UNACCEPTABLE  # unacceptable.


class TwoSidedFunction(ClassFunction):
    """Class function for value or range is better."""

    def __init__(self, lower_bounds, upper_bounds):
        """Construct class function."""
        # requires two sets of bounds for lower and upper limit
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

    def evaluate(self, g):
        """Evaluate the class function against some input."""
        raise NotImplementedError

    def build_splines(self, nsc, alpha=0.05, initial_beta=1.5):
        """Compute the splines that define this class function's values."""
        raise NotImplementedError


class ValueIsBetter(TwoSidedFunction, SmoothClassFunction):
    """A certain value is better (3-S)."""

    def evaluate(self, g):
        """Evaluate the class function against some input."""
        raise NotImplementedError

    def build_splines(self, nsc, alpha=0.05, initial_beta=1.5):
        """Compute the splines that define this class function's values."""
        raise NotImplementedError

    def which_region(self, g):
        """Determine which region dependent variable value g is in."""
        raise NotImplementedError


class HardClassFunction(ClassFunction):
    """
    Hard constraint: Must be at least/most x.

    These are simply turned into pure constraints rather than adding
    them to the aggregate objective function in any functional form.
    """

    def evaluate(self, g):
        """Do nothing because this is converted to a constraint."""
        pass

    def build_splines(self, nsc, alpha=0.05, initial_beta=1.5):
        """Do nothing because this is converted to a constraint."""
        pass


class MustBeAbove(HardClassFunction):
    """Value must be above a bound (1-H)."""

    def acceptability(self, g):
        """
        Rank the acceptability of value g on scale from 0 to 1.

        Useful for plotting relative changes to values.
        """
        if g >= self.bounds.cutoff:
            return 1.0
        return 0.0


class MustBeBelow(HardClassFunction):
    """Value must be below a bound (2-H)."""

    def acceptability(self, g):
        """
        Rank the acceptability of value g on scale from 0 to 1.

        Useful for plotting relative changes to values.
        """
        if g <= self.bounds.cutoff:
            return 1.0
        return 0.0


class MustBeInRange(TwoSidedFunction, HardClassFunction):
    """Value must be between two bounds (3-H)."""

    pass


def from_input(filename):
    """Build class functions defined in an input file."""
    funcs = collections.OrderedDict()  # order matters.
    with open(filename) as inp:
        userinp = yaml.safe_load(inp)
        for details in userinp["dependents"]:
            dependent_name = details["name"]
            print(f"Loading {dependent_name}")
            cls = getattr(sys.modules[__name__], details["class"], None)
            if not (isinstance(cls, type) and issubclass(cls, ClassFunction)):
                raise ValueError(
                    f"Unknown class function {details['class']!r} "
                    f"for {dependent_name!r}"
                )
            if issubclass(cls, SmoothClassFunction):
                func = cls(SoftBounds(*details["bounds"]))
            else:
                func = cls(HardBounds(details["bound"]))
            func.name = dependent_name
            func.units = details.get("units")
            funcs[dependent_name] = func
    build_all_splines(funcs.values())
    return funcs


def build_all_splines(functions):
    """
    Perform iteration to define splines.

    This loops over all soft constraints until the value of beta converges.
    """
    softfuncs = [
        func for func in functions if issubclass(func.__class__, SmoothClassFunction)
    ]
    nsc = len(softfuncs)
    max_beta = 1.5
    last_beta = max_beta
    for _i in range(MAX_ITERATIONS):
        print(f"beta is {max_beta}")
        for func in softfuncs:
            new_beta = func.build_splines(
                nsc, initial_beta=max_beta
            )  # returns beta required for convex
            if not new_beta:
                raise RuntimeError(f"Error building {func}. Beta did not converge.")
            max_beta = max(max_beta, new_beta)

        if max_beta == last_beta:
            # success.
            break
        last_beta = max_beta
    else:
        raise RuntimeError("PP algorithm to build class functions did not converge")

    print(f"Successful build of {nsc} PP class functions")
