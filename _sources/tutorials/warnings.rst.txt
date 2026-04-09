Warnings
========

In order to make debugging easier, ``powerlaw`` tries to give warnings when
abnormal cases are encountered during fitting, distribution comparisons, and
other functions. This page describes each warning, and ways you might be
able to fix it, or when it is appropriate to ignore the warning.

.. warning::

    ``discrete=True but data does not exclusively contain integer values. Casting to integer...``

``powerlaw`` has special options for discrete distributions, and this warning
is raised in ``Distribution.__init__()`` if the user has specified that the distribution is discrete, yet the
data based in the ``data`` keyword contained float (or other non-integer)
types.

The easiest fix for this is just to cast your data to integer before trying
to fit a distribution, assuming you are indeed trying to fit a discrete
distribution. Otherwise, make sure you pass ``discrete=False`` (which is the
default) to ``Fit`` or distribution objects.

-----

.. warning::
    
    ``discrete=False but data exclusively contains integer values. Consider using discrete=True.``


``powerlaw`` has special options for discrete distributions, and this warning
is raised in ``Distribution.__init__()`` if the user has not specified that the distribution is discrete, yet the
data based in the ``data`` keyword contains only integer types.

This doesn't necessarily mean something is wrong: for example, there could
be some experimental limitation (precision of a sensor, etc.) that only
lets you measure a continuous quantity in discrete steps. As long as the
underlying quantity is truly continuous, you can safely ignore this warning.
Otherwise, make sure you pass ``discrete=True`` to ``Fit`` or distribution
objects.

----

.. warning::

    ``No valid fits found for distribution <name>.``

This warning is raised in ``Distribution.fit()`` when numerical fitting
fails to find a set of good parameters values, and/or they are out of range.

This could be due to parameter constraints, or your initial condition, or
simply that the data is vastly different from the distribution you tried to
fit. It may help to examine the initial condition and make sure it is
reasonable:

.. code-block::

    # Assuming this raises the warning
    fit = powerlaw.Fit(data, ...)

    print(fit.<dist_name>.initial_condition)

    # Or if just directly from a distribution, eg. a power law
    dist = powerlaw.Power_Law(data, ...)
    print(dist.initial_condition)


----

.. warning::

    ``Fitted parameters are very close to the edge of parameter ranges; consider changing these ranges.``

This warning is raised in ``Distribution.fit()`` if the numerical fitting
results in one or more parameters very close to the edge of the parameter
range.

This often means that the range is too restrictive, and should be expanded
to encapsulate the proper solution. You can examine the parameters and their
bounds with:

.. code-block::

    # For a power law (can easily replace with another distribution)
    fit = powerlaw.Power_Law(data, ...)

    print(fit.parameters, fit.parameter_ranges)

----

.. warning::

    ``Not enough data to compute distance metrics like Kolmogorov-Smirnov distance, returning nan.``

This warning is raised in ``Distribution.compute_distance_metrics()`` and
means that there are less than 2 samples from the original ``data`` that
are contained in the domain of the distribution, ie. ``[xmin, xmax]``.

Likely this means that fitting ``xmin`` didn't work, or the explicit value
for ``xmin`` or ``xmax`` is far too large or too small, respectively.

----

.. warning::

    ``Likely underflow or overflow error: the optimal fit for this distribution gives values that are so extreme that we lack the numerical precision to calculate them.``

This warning is raised in ``Distribution.cdf()`` when there are nan values
in the array of CDF values.


----

.. warning:: ``Power law distributions with alpha close to 1 without an xmax can be very noisy; it is recommended to give some xmax.``

This warning is raised in ``powerlaw.Power_Law.__init__()`` when you have an
exponent alpha that is in the range ``[1, 1.1]``. 

Power laws in this regime can be very noisy due to the discontinuity at
:math:`\alpha = 1`, which is especially the case when there is no value given
for ``xmax``. For these unbounded power laws, you can easily spread your
data across more than 200+ decades, which obviously is not ideal when you
only want to sample on the order of 1000s of values.

The easiest way to fix this is just by providing an ``xmax`` value, where the
easiest value to use without change the fit properties is just ``np.max(data)``.

----

.. warning:: ``Distribution with alpha <= 1 has no xmax; setting xmax to be max(data) otherwise cannot continue. Consider setting an explicit value for xmax.``

This warning is raised in ``Distribution._pdf_continuous_normalizer`` when
the use tries to normalize a power law distribution with :math:`\alpha <= 1`
and no explicit ``xmax``.

Power law distributions with :math:`\alpha <= 1` are not normalizable for
an unbounded domain, so the only way to compute the normalization constant
is by assuming that the distribution is bounded at the edge of the data.
It is much better practice to actually give an explicit value for ``xmax``.

If the value of alpha should not be less than 1, you should look into changing
your parameter ranges and constraints to make sure that the fitting succeeds.

----

.. warning:: ``Values less than or equal to 0 in data. Throwing out 0 or negative values.``

This warning is raised upon creating a ``Fit`` object (in ``Fit.__init__()``)
when the provided ``data`` includes values that have values less than or
equal to zero.

Zero or negative values are invalid for the heavy-tailed probability distributions
implemented in the library, so must be removed. It is better to filter
these before passing the ``data`` to the ``Fit`` class for clarity.

----

.. warning:: ``Less than 2 unique data values for fitting xmin! Returning nans.``

This error is raised in ``Fit.find_xmin()`` when you have two or fewer data
points.

Besides giving you only two possible values of ``xmin``, trying to fit a
distribution when you only have two data points is meaningless. Make sure
that your ``xmax`` isn't removing the majority of your data, and generally
check the PDF of your data.

----

.. warning:: ``No valid values for xmin found.``

This warning is raised in :meth:`Fit.find_xmin` when none of the possible
``xmin`` values yielded a successful fit.

Consider changing you parameter ranges, constraints, sigma constraints, etc.
