import unittest
from collections import namedtuple
import math
import os

from physprog import classfunctions
from physprog import optimize


THIS_DIR = os.path.dirname(__file__)

SAMPLE_INPUT = os.path.join(THIS_DIR, 'sample-input.yaml')

class TestInput(unittest.TestCase):
    def test_read_class_functions(self):
        functions = classfunctions.from_input(SAMPLE_INPUT)
        self.assertTrue('frequency' in functions)

class Test_Sample_Problem(unittest.TestCase):
    def test_optimization(self):
        beam = SampleProblemBeam()
        # check initial conditions
        self.assertAlmostEqual(beam.frequency(), 113.0, delta=0.5)
        self.assertAlmostEqual(beam.cost(), 1060.0)
        self.assertAlmostEqual(beam.mass(), 2230.0)
        
        prefs = classfunctions.from_input(SAMPLE_INPUT)
        optimize.optimize(beam, prefs)

        # not rigorous, but happens in this problem
        self.assertLess(beam.cost(), 1060.0)


SampleDesign = namedtuple('SampleDesign', ['d1', 'd2', 'd3', 'b' , 'L'])

class SampleProblemBeam(object):
    """Sample beam design problem from Messac, 1996."""
    E1 = 1.6e9
    C1 = 500.0
    RHO1 = 100.0
    E2 = 70e9
    C2 = 1500.0
    RHO2 = 2770.0
    E3 = 200e9
    C3 = 800.0
    RHO3= 7780.0
    
    def __init__(self):
        self._design = SampleDesign(0.3, 0.35, 0.40, 0.40, 5.0)  # initial

    @property
    def design(self):
        return self._design

    @design.setter
    def design(self, val):
        self._design = SampleDesign(*val)

    def analyze(self):
        """Convert input design into output design parameters."""
        return [self.frequency(), self.cost(), self.width(), self.length(), 
                self.mass(), self.semiheight(),self.width_layer1(), 
                self.width_layer2(), self.width_layer3()]
        
    @property
    def ei(self):
        ds = self.design
        return 2.0 / 3.0 * ds.b * (self.E1 * ds.d1 ** 3 +
                                   self.E2 * (ds.d2 ** 3 - ds.d1 ** 3) +
                                   self.E3 * (ds.d3 ** 3 - ds.d2 ** 3))
    @property
    def mu(self):
        ds = self.design
        return 2 * ds.b * (self.RHO1 * ds.d1 +
                           self.RHO2 * (ds.d2 - ds.d1) +
                           self.RHO3 * (ds.d3 - ds.d2))

    def frequency(self):
        return math.pi / (2 * self.design.L ** 2) * math.sqrt(self.ei / self.mu)

    def cost(self):
        ds = self.design
        # cost in the paper says 1060 but I'm getting 212, exactly a
        # factor of 5 off. But why?? Ah, because cost should have L in it!
        # That's a typo in the paper.
        return 2 * ds.b * ds.L * (self.C1 * ds.d1 +
                                  self.C2 * (ds.d2 - ds.d1) +
                                  self.C3 * (ds.d3 - ds.d2))

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


if __name__ == '__main__':
    unittest.main()
