# PhysProg

[![Build Status](https://travis-ci.org/partofthething/physprog.svg?branch=master)](https://travis-ci.org/partofthething/physprog)

**physprog** ("Physical Programming") is a tool for people who have something 
with multiple metrics they want to optimize simultaneously. You may be an 
engineer wanting to minimize the cost, minimize the weight, and maximize the 
width of a widget. Or you may be a fund manager trying to maximize
your up-market capture ratio while minimizing your down-market capture ratio. 

Regardless of what you want to optimize, it's often a dark art to weight
multiple performance metrics into one aggregate objective function to be passed
off to a general purpose function minimizer. 

**physprog** eliminates the guess-work by taking meaningful preferences for each 
performance metric and building the 
aggregate objective function for you. With this, you can focus your time on 
adjusting the design of your product and less time (no time, really)
adjusting the weights so that they accurately reflect your preferences. 

**physprog** is an implementation of the Messac's Physical Programming algorithm 
[[1](https://messac.expressions.syr.edu/wp-content/uploads/2012/05/Messac_1996_AIAA_PP.pdf)].
For each performance metric, you express your preferences by defining the
boundaries between 6 preference regions: 

* Highly Desirable
* Desirable
* Tolerable
* Undesirable
* Highly Undesirable
* Unacceptable

You can choose between several *class functions* to reflect these
preferences, such as:

* Smaller is better
* Larger is better
* Value is better
* Range is better

Of course, you can also define hard constraints for
performance metrics, as in: "thickness has to be at least 1mm."

**physprog** can just hand you the objective function for you to pass on
to whatever optimization routine you prefer, or it has some basic
code to pass it through [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize)
and do the full optimization for you.

The code for this project, as well as the issue tracker, etc. is
[hosted on GitHub](https://github.com/partofthething/physprog)

# Installing it

Directly from source:

    git clone git@github.com:partofthething/physprog.git
    cd physprog
    python setup.py install
    

# Using it

To use **physprog** you need two things: your preferences and a model. 

## Specifying Preferences

You specify your preferences in a YAML input file. The bounds input
should go from most desirable to least desirable, where anything
beyond the last bound is unacceptable. Here's an example:

```yaml
dependents:
  - name: frequency
    class: LargerBetter
    units: Hz
    bounds:
       - 200.0
       - 150.0
       - 120.0
       - 110.0
       - 100.0
  - name: cost
    class: SmallerBetter
    units: kg
    bounds:
       - 1000.0
       - 1800.0
       - 1900.0
       - 1950.0
       - 2000.0
  - name: width
    class: MustBeAbove
    units: m
    bound: 0.01
```

NOTE: You can also specify preferences in a dictionary and bypass
the input file.  

## Specifying a model

Your model must take inputs and produce outputs. Inputs should be 
defined in a tuple or list in the same order that your preferences were 
specified in. Outputs corresponding to inputs should be computed 
by an `evaluate` method. There should be a method named after each of 
your dependent variables. With the example preferences above, your model 
may look like:

```python
SampleDesign = namedtuple('SampleDesign', ['d1', 'd2', 'd3', 'b', 'L'])

class SampleProblemBeam(object):
    """Dummy sample problem."""

    def __init__(self):
        self._design = SampleDesign(0.3, 0.35, 0.40, 0.40, 5.0)  # initial

    def evaluate(self, x=None):
        """Convert input design into output design parameters."""
        if x is not None:
            self.design = x
        return [self.frequency(), self.cost(), self.width()]

    def frequency(self):
        return math.pi / (2 * self.design.L ** 2) * self.design.d2

    def cost(self):
        ds = self.design
        return 2 * ds.b * ds.L 

    def width(self):
        return self.design.b
```

A complete sample problem involving the design of a beam is [included
in the tests](./physprog/tests/test_sample_problem.py). 


## Running an optimization problem

To compute the optimal design of your thing, 
do something like this:

```python
from physprog import classfunctions, optimize
preferences = classfunctions.from_input('input.yaml')
model = SampleProblemBeam()
optimize.optimize(model, preferences, plot=True)
```
The results of the full sample problem are shown below:

![Picture of optimization results](./assets/sample-results.png "Sample problem results")

## Building the objective function

If you just want an aggregate objective function without optimizing,
you can do this:

```python
from physprog import classfunctions, objective
preferences = classfunctions.from_input('input.yaml')
model = SampleProblemBeam()
aggregate = objective.build_objective(model, preferences)
```

Then you can pass the `aggregate` to any other optimization engine. 

# Contributing

You are encouraged to make contributions to make this system more 
useful to everyone. There are automated tests that run with the 
`tox` command so make sure those all pass (including style checks) 
before submitting a PR. If you have a big change in mind you
may want to contact the author 

# License

This package is released under the Apache-2.0 license [reproduced
here](./LICENSE).

# References and See Also

1. [Messac, Achille. "Physical programming-effective optimization for computational design." AIAA journal 34.1 (1996): 149-158.](https://messac.expressions.syr.edu/wp-content/uploads/2012/05/Messac_1996_AIAA_PP.pdf)
2. [Multi-objective optimization (Wikipedia)](https://en.wikipedia.org/wiki/Multiobjective_optimization)