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
RangeBounds = collections.namedtuple("RangeBounds", ["lower", "upper"])

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

# Default ValueIsBetter.pull: fraction of the edge value the target dips below
# the edges of the highly desirable region.
DEFAULT_PULL = 0.5

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

    # +1 if bounds increase from best to worst, -1 if they decrease.
    direction = None

    # If True, the class function has zero slope at the first bound so it
    # can join a flat "highly desirable" region smoothly (range is better).
    flat_best_region = False

    def __init__(self, bounds):
        """Construct class function."""
        ClassFunction.__init__(self, bounds)
        self._check_bounds()
        self.slopes = []
        self.region_slopes = []
        self.gis = []
        self.dgis = []

    def _check_bounds(self):
        """Make sure bounds run strictly from most to least desirable."""
        if self.direction is None:
            return
        steps = np.diff(self.bounds) * self.direction
        if len(self.bounds) != len(SoftBounds._fields) or np.any(steps <= 0):
            order = "increasing" if self.direction > 0 else "decreasing"
            raise ValueError(
                f"{type(self).__name__} needs {len(SoftBounds._fields)} strictly "
                f"{order} bounds (most to least desirable), got {tuple(self.bounds)}"
            )

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
                # A flat best region needs zero slope here to join smoothly.
                if self.flat_best_region:
                    self.slopes[i] = 0.0
                else:
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

    direction = +1

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

    direction = -1

    def which_region(self, g):
        """Determine which region dependent variable value g is in."""
        high = +float("inf")
        for i, lower_edge in enumerate(self.bounds):
            if lower_edge <= g <= high:
                # this is our region.
                return i
            high = lower_edge

        return UNACCEPTABLE  # unacceptable.


class TwoSidedFunction(SmoothClassFunction):
    """
    Base for soft preferences that are bad on both sides (3-S and 4-S).

    It is built from two one-sided class functions: a larger-is-better
    function for values below the best region and a smaller-is-better
    function for values above it. The best ("highly desirable") region is
    the interval between the two first bounds, which is handled by
    subclasses.

    Parameters
    ----------
    lower_bounds : sequence of 5 floats
        Bounds below the best region, from most to least desirable, so they
        decrease (e.g. ``[4.5, 4.0, 3.5, 3.0, 2.5]``).
    upper_bounds : sequence of 5 floats
        Bounds above the best region, from most to least desirable, so they
        increase (e.g. ``[5.5, 6.0, 6.5, 7.0, 7.5]``).
    """

    def __init__(self, lower_bounds, upper_bounds):
        """Construct class function."""
        self.lower = LargerBetter(SoftBounds(*lower_bounds))
        self.upper = SmallerBetter(SoftBounds(*upper_bounds))
        self.lower.flat_best_region = self.upper.flat_best_region = (
            self.flat_best_region
        )
        if not self.lower.bounds[0] < self.upper.bounds[0]:
            raise ValueError(
                f"{type(self).__name__} needs the first lower bound "
                f"({self.lower.bounds[0]}) below the first upper bound "
                f"({self.upper.bounds[0]})"
            )

    @property
    def bounds(self):
        """All bounds in increasing order (lower side, then upper side)."""
        return (*reversed(self.lower.bounds), *self.upper.bounds)

    @property
    def gis(self):
        """Class function values at the bounds (the same on both sides)."""
        return self.upper.gis

    @property
    def best_region(self):
        """The (low, high) edges of the highly desirable region."""
        return self.lower.bounds[0], self.upper.bounds[0]

    def side(self, g):
        """Return the one-sided function governing ``g``, or None if best."""
        low, high = self.best_region
        if g < low:
            return self.lower
        if g > high:
            return self.upper
        return None

    def which_region(self, g):
        """Determine which region dependent variable value g is in."""
        side = self.side(g)
        return AWESOME if side is None else side.which_region(g)

    def evaluate(self, g):
        """Evaluate the class function against some input."""
        side = self.side(g)
        if side is None:
            return self.evaluate_best_region(g)
        return side.evaluate(g)

    def evaluate_best_region(self, g):
        """Evaluate the class function inside the highly desirable region."""
        raise NotImplementedError

    def acceptability(self, g):
        """Rank the acceptability of value g on scale from 0 to 1."""
        side = self.side(g)
        if side is None:
            return 0.9
        return side.acceptability(g)

    def build_splines(self, nsc, alpha=0.05, initial_beta=1.5):
        """Build both sides with a common beta so their values match."""
        beta = initial_beta
        for _i in range(MAX_ITERATIONS):
            new_beta = max(
                self.lower.build_splines(nsc, alpha, beta),
                self.upper.build_splines(nsc, alpha, beta),
            )
            if new_beta == beta:
                self._finish_build()
                return beta
            beta = new_beta
        raise RuntimeError("Two-sided class function construction did not converge.")

    def _finish_build(self):
        """Set up the best region once both sides are built (subclass hook)."""


class ValueIsBetter(TwoSidedFunction):
    """
    A certain value is better (3-S).

    The class function has its minimum at ``target`` and rises smoothly
    toward both edges of the highly desirable region, joining the
    one-sided splines with matching value and slope. If ``target`` is not
    given, it is the middle of the highly desirable region.

    ``pull`` sets how strongly the target attracts the design: the minimum
    sits ``pull`` times the edge value below the edges, so 0 is nearly flat
    (any value in the region is about as good) and values near 1 make the
    target worth almost as much as the whole highly desirable region of
    another metric.
    """

    def __init__(self, lower_bounds, upper_bounds, target=None, pull=DEFAULT_PULL):
        """Construct class function."""
        TwoSidedFunction.__init__(self, lower_bounds, upper_bounds)
        low, high = self.best_region
        if target is None:
            target = (low + high) / 2.0
        if not low < target < high:
            raise ValueError(
                f"target {target} must lie strictly between {low} and {high}"
            )
        if not 0.0 <= pull < 1.0:
            raise ValueError(f"pull must be in [0, 1), got {pull}")
        self.target = target
        self.pull = pull
        self.depth = 0.0

    def _finish_build(self):
        """
        Set how far the minimum dips below the edge value.

        Each half is a cubic Hermite spline from the edge (value g0, slope s)
        to the target (value g0 - depth, slope 0). It decreases monotonically
        toward the target as long as ``depth >= s*h/3``, where h is the
        half's width, so the requested pull is raised to that if needed.
        """
        low, high = self.best_region
        g0 = self.gis[0]
        s_low = -self.lower.slopes[0]
        s_high = self.upper.slopes[0]
        h_low, h_high = self.target - low, high - self.target
        monotonic_min = max(s_low * h_low, s_high * h_high) / 3.0
        self.depth = max(self.pull * g0, monotonic_min)
        if self.depth >= g0:
            raise ValueError(
                f"Highly desirable region ({low}, {high}) of {self.name or 'value'} "
                "is too wide relative to the desirable regions to build a "
                "positive class function; widen the desirable bounds"
            )

    def evaluate_best_region(self, g):
        """Smooth bowl with its minimum at the target."""
        low, high = self.best_region
        g0 = self.gis[0]
        bottom = g0 - self.depth
        if g <= self.target:
            return _hermite(
                g, (low, g0, self.lower.slopes[0]), (self.target, bottom, 0)
            )
        return _hermite(g, (self.target, bottom, 0), (high, g0, self.upper.slopes[0]))

    def acceptability(self, g):
        """Rank acceptability; 1 at the target falling to 0.8 at the edges."""
        if self.side(g) is not None:
            return TwoSidedFunction.acceptability(self, g)
        low, high = self.best_region
        edge = low if g <= self.target else high
        return 1.0 - 0.2 * (g - self.target) / (edge - self.target)


class RangeIsBetter(TwoSidedFunction):
    """
    Any value within a range is equally good (4-S).

    The class function is flat across the highly desirable region and
    rises smoothly outside it.
    """

    flat_best_region = True

    def evaluate_best_region(self, g):
        """Return the constant value of the highly desirable region."""
        return self.gis[0]


def _hermite(x, start, end):
    """Evaluate the cubic Hermite spline between (x, y, slope) end points."""
    (x0, y0, m0), (x1, y1, m1) = start, end
    h = x1 - x0
    u = (x - x0) / h
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


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


class MustBeInRange(HardClassFunction):
    """Value must be between two bounds (3-H). Bounds are a RangeBounds."""

    def __init__(self, bounds):
        """Construct class function."""
        bounds = RangeBounds(*bounds)
        if not bounds.lower < bounds.upper:
            raise ValueError(f"MustBeInRange needs lower < upper, got {bounds}")
        HardClassFunction.__init__(self, bounds)

    def acceptability(self, g):
        """
        Rank the acceptability of value g on scale from 0 to 1.

        Useful for plotting relative changes to values.
        """
        if self.bounds.lower <= g <= self.bounds.upper:
            return 1.0
        return 0.0


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
            if issubclass(cls, ValueIsBetter):
                func = cls(
                    details["lower_bounds"],
                    details["upper_bounds"],
                    target=details.get("target"),
                    pull=details.get("pull", DEFAULT_PULL),
                )
            elif issubclass(cls, TwoSidedFunction):
                func = cls(details["lower_bounds"], details["upper_bounds"])
            elif issubclass(cls, SmoothClassFunction):
                func = cls(SoftBounds(*details["bounds"]))
            elif issubclass(cls, MustBeInRange):
                func = cls(RangeBounds(*details["bounds"]))
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
