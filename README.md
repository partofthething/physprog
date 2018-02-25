# PhysProg


.. image:: https://travis-ci.org/partofthething/physprog.svg?branch=master
    :target: https://travis-ci.org/partofthething/physprog
    
When you have a product you want to optimize, it's often a dark-art to weight
multiple design performance metrics into one aggregate objective function to 
minimize. **physprog** eliminates the guess-work by accepting physically-
meaningful preferences for each design metric and then building the 
aggregate objective function for you. With this, you can focus more time on 
adjusting the design of your product and less time (no time, really)
adjusting the weights so that they reflect your preferences. 

**physprog** is an implementation of the Physical Programming algorithm [Messac96]_,
which takes the guess-work out of building aggregate objective functions on
their way to multiobjective optimization algorithms. 

The code for this project, as well as the issue tracker, etc. is
[hosted on GitHub](https://github.com/partofthething/physprog)

# Installing it

Directly from source:

    git clone git@github.com:partofthething/physprog.git
    cd physprog
    python setup.py install
    

# Using it



# Contributing

   
# License

This package is released under the MIT License, [reproduced
here](https://github.com/partofthething/physprog/blob/master/LICENSE).

# References

.. [Messac96] Messac, Achille. "Physical programming-effective optimization for computational design." AIAA journal 34.1 (1996): 149-158. [Link](https://messac.expressions.syr.edu/wp-content/uploads/2012/05/Messac_1996_AIAA_PP.pdf)