"""Definitions of the various kinds of Class Functions."""

import collections
import math
import sys

import yaml
import numpy as np
import matplotlib.pyplot as plt


# data structure representing the limits of each range.
# right-hand limit is either positive or negative infinity.
# we order them consistently with the region numbering, so
# the most desirable regions are first.
SoftBounds = collections.namedtuple(
    'SoftBounds',
    ['awesome', 'desirable', 'tolerable', 'undesirable', 'horrible'])
HardBounds = collections.namedtuple('HardBounds', ['cutoff'])

# region labels
AWESOME = 0
DESIRABLE = 1
TOLERABLE = 2
UNDESIRABLE = 3
HORRIBLE = 4
UNACCEPTABLE = 5

MAX_ITERATIONS = 20


class ClassFunction(object):
    """Basic class function expressing preferences."""

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
            # in first region we just use a simple exponential curve.
            return self.gis[0] * (math.exp(
                self.slopes[1] / self.gis[0] * g - self.bounds.desirable))
        elif i == UNACCEPTABLE:
            # super steep outside acceptable bounds
            # NOTE: could also set these as inequality constraints
            return abs(self.gis[-1] * 50 * g)
        a, b, c, d = self.evaluate_spline_coeffs(i)
        xi, width = self.get_region_fraction(g, i)
        xi4 = xi ** 4
        xim14 = (xi - 1) ** 4

        return width ** 4 * (
            a / 12.0 * xi4 + b / 12.0 * xim14) + c * width * xi + d

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
        betas = np.arange(initial_beta, int(MAX_ITERATIONS / 0.5), 0.5)
        for beta in betas:
            self.gis = []
            self.dgis = []
            self.slopes = [0.0 for _bi in self.bounds]
            self.region_slopes = []
            self._compute_region_slopes(nsc, beta)
            acceptable = self._compute_pointwise_slopes(alpha)
            if acceptable:
                return beta

        raise RuntimeError('Class function construction did not converge.')

    def _compute_region_slopes(self, nsc, beta):
        """Perform first loop in spline-building iteration."""
        for i, _bpv in enumerate(self.bounds):
            if i == 0:
                gi = dgi = 0.1
                region_slope = 0.0
            else:
                dgi = beta * nsc * dgi
                gi = gi + dgi
                region_slope = dgi / (
                    self.bounds[i] - self.bounds[i - 1])
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
                self.slopes[i] = alpha * region_slope
            else:
                # slope[k]=skMin+alpha*(-skMax+skMin)
                # skMin-skMax in the paper. But if you calculate it you see
                # the paper is wrong, but the expression is right.
                slope_min_i = (
                    4.0 * region_slope - self.slopes[i - 1]) / 3.0
                slope_max_i = 4.0 * region_slope - 3 * self.slopes[i - 1]
                self.slopes[
                    i] = slope_min_i + alpha * (slope_max_i - slope_min_i)

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

        a = (3.0 * (3.0 * si + sim1) - 12.0 * rsi) / (2.0 * width ** 3)
        b = (12.0 * rsi - 3.0 * (si + 3.0 * sim1)) / (2.0 * width ** 3)
        c = 2 * rsi - 0.5 * si - 0.5 * sim1
        d = gim1 - width * (0.5 * rsi - 0.125 * si - 0.375 * sim1)

        return a, b, c, d

    def which_region(self, g):
        """Determine which region the value g is in."""
        raise NotImplementedError

    def plot(self, fname=None):
        """Plot this class function."""
        x = np.linspace(self.bounds[0], self.bounds[-1], 200)
        y = [self.evaluate(xi) for xi in x]
        plt.figure()
        plt.plot(x, y, label='Class func')
        plt.plot(self.bounds, self.gis, 'o', label='Algorithm')
        plt.grid(color='0.7')
        plt.xlabel('Dependent Variable')
        plt.ylabel('PP Transformed Class Value')
        if fname:
            plt.savefig(fname)
        else:
            plt.show()

    def acceptability(self, g):
        """
        Rank the acceptability of value g on scale from 0 to 1.

        Useful for plotting relative changes to values.
        """
        region = self.which_region(g)
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
        low = -float('inf')
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
        high = +float('inf')
        for i, lower_edge in enumerate(self.bounds):
            if lower_edge <= g <= high:
                # this is our region.
                return i
            high = lower_edge

        return UNACCEPTABLE  # unacceptable.


class TwoSidedFunction(ClassFunction):
    """Class function for value or range is better."""

    def __init__(self, lower_bounds, upper_bounds):  # pylint: disable=super-init-not-called
        """Construct class function."""
        # requires two sets of bounds for lower and upper limit
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

    def evaluate(self, g):
        """Evaluate the class function against some input."""
        return NotImplementedError

    def build_splines(self, nsc, alpha=0.05, initial_beta=1.5):
        """Compute the splines that define this class function's values."""
        return NotImplementedError


class ValueIsBetter(TwoSidedFunction, SmoothClassFunction):
    """A certain value is better (3-S)."""

    def evaluate(self, g):
        """Evaluate the class function against some input."""
        return NotImplementedError

    def build_splines(self, nsc, alpha=0.05, initial_beta=1.5):
        """Compute the splines that define this class function's values."""
        return NotImplementedError

    def which_region(self, g):
        """Determine which region dependent variable value g is in."""
        return NotImplementedError


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
        userinp = yaml.load(inp)
        for details in userinp['dependents']:
            dependent_name = details['name']
            print('Loading {}'.format(dependent_name))
            cls = getattr(sys.modules[__name__], details['class'])
            if issubclass(cls, SmoothClassFunction):
                bounds = SoftBounds(*details['bounds'])
                funcs[dependent_name] = cls(bounds)
            else:
                bound = HardBounds(details['bound'])
                funcs[dependent_name] = cls(bound)
    build_all_splines(funcs.values())
    return funcs


def build_all_splines(functions):
    """
    Perform iteration to define splines.

    This loops over all soft constraints until the value of beta converges.
    """
    softfuncs = [
        func for func in functions
        if issubclass(func.__class__, SmoothClassFunction)
    ]
    nsc = len(softfuncs)
    max_beta = 1.5
    last_beta = max_beta
    for _i in range(MAX_ITERATIONS):
        print('beta is {0}'.format(max_beta))
        for func in softfuncs:
            new_beta = func.build_splines(
                nsc, initial_beta=max_beta)  # returns beta required for convex
            if not new_beta:
                raise RuntimeError(
                    'Error building {0}. Beta did not converge.'.format(func))
            max_beta = max(max_beta, new_beta)

        if max_beta == last_beta:
            # success.
            break
        last_beta = max_beta
    else:
        raise RuntimeError(
            'PP algorithm to build class functions did not converge')

    print('Successful build of {0} PP class functions'.format(nsc))
