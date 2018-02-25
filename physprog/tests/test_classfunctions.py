import unittest


from physprog import classfunctions


class TestSmallerBetter(unittest.TestCase):

    def setUp(self):
        self.bounds = classfunctions.SoftBounds(10, 20, 30, 40, 50)
        self.sb = classfunctions.SmallerBetter(self.bounds)

    def test_which_region(self):
        self.assertEqual(self.sb.which_region(0), 0)
        self.assertEqual(self.sb.which_region(35), 3)
        self.assertIs(self.sb.which_region(55), 5)


class TestLargerBetter(unittest.TestCase):

    def setUp(self):
        self.bounds = classfunctions.SoftBounds(50, 40, 30, 20, 10)
        self.lb = classfunctions.LargerBetter(self.bounds)

    def test_which_region(self):
        self.assertEqual(self.lb.which_region(55), 0)
        self.assertEqual(self.lb.which_region(35), 2)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
