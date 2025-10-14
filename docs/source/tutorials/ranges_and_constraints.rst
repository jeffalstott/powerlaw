Parameter Ranges and Constraints
================================

When performing a fit to a probability distribution, you may want to
restrict the parameter space that must be explored for the fit.
``powerlaw`` provides two ways to do this: parameter ranges, which give
fixed bounds for parameter values, and constraints, which are relational
expressions between parameters.

If neither of these are provided, the fitting will be performed with no
constraints, and the default bounds for parameters defined in each
distribution's ``DEFAULT_PARAMETER_RANGES`` property.

.. code-block::

    data = powerlaw.load_test_dataset('blackouts')

    pl = powerlaw.Power_Law(data)

For a power law, the default parameter ranges are:

.. code-block::

    powerlaw.Power_Law.DEFAULT_PARAMETER_RANGES

.. code-block:: output

    {'alpha': [0, 3]}

If you want to manually set this parameter range, you can easily do so:

.. code-block::

    # Some data that has an exponent between 3 and 4
    data = ...

    pl = powerlaw.Power_Law(data, parameter_ranges={"alpha": [3, 4]})

You can do the same thing with the ``Fit`` class:

.. code-block::

    # Some data that has an exponent between 3 and 4
    data = ...

    fit = powerlaw.Fit(data, parameter_ranges={"alpha": [3, 4]})

Of course, this parameter range will only apply to distributions that have
a parameter called ``"alpha"``; ``fit.exponential``, for example,
would be unaffected.

For a distribution with multiple parameters, the format is exactly the same:

.. code-block::

    tpl = powerlaw.Truncated_Power_Law(data,
                                       parameter_ranges={"alpha": [3, 4]}, "Lambda": [0, 7]})


Let's say instead of using fixed bounds for these parameters, you wanted to
specify a relational constraint between them, for example, that ``Lambda``
is always less than ``alpha``. This can be accomplished by passing
constraint functions using the ``parameter_constraints`` keyword to either
``Fit`` or a ``Distribution`` object.

The argument of such a function should be a tuple.
.. code-block::
    
    def constraint(params):
        p1, p2 = params
