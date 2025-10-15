Loading data
============

This library is based on analyzing probability distributions, which should
be given as a collection of values sampled from the distribution. Note that
this data should not be binned or anything prior to passing to any functions
or classes.

Loading data to be used in this package is very straight forward, since you
only need a numpy array or python list:

.. code-block::

    import numpy as np
    import powerlaw

    data = np.genfromtxt('some_data.csv')

    # Now we can use any powerlaw function or class
    fit = powerlaw.Fit(data=data, ...)

If you want to explore the functionality provided by ``powerlaw`` but don't
have your own data, you can load any of the test datasets using the
:meth:`~powerlaw.load_test_dataset` method.

.. code-block::
    
    import powerlaw

    data = powerlaw.load_test_dataset('blackouts')


The datasets listed below are available; this information can be obtained
using :meth:`powerlaw.print_test_datasets`:

.. list-table:: Available test datasets
    :widths: 10 40 8 5
    :header-rows: 1

    * - Dataset name
      - Description
      - Type
      - Source

    * - blackouts
      - Number of people (in thousands) in the United States affected by electricity blackouts between 1984 and 2002.
      - Continuous
      - [1]

    * - cities
      -
      - Continuous
      - 

    * - fires
      -
      - Continuous
      - 

    * - flares
      -
      - Continuous
      - 

    * - quakes
      -
      - Continuous
      - 

    * - surnames
      -
      - Continuous
      - 

    * - terrorism
      -
      - Discrete
      - 

    * - words
      - Frequency of word usage in Herman Melville’s novel "Moby Dick"
      - Discrete
      - [1]

.. note::

    Most of these test files were copied from ``plfit`` which were in turn
    copied from ``agpy``, so I'm not sure of the original source.

.. figure:: ../images/test_datasets.png

    PDF for all of the test datasets, generated with :meth:`powerlaw.plot_test_datasets`.

References
----------

[1] Newman MEJ (2005) Power laws, Pareto distributions and Zipfs law 46
