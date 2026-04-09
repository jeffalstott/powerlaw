API
===

.. toctree::
    :maxdepth: 1

    fit
    distributions

The intended front-end usage of this library involves creating :class:`Fit`
objects, and then accessing various distributions as properties of that
object.

.. autosummary::

    powerlaw.Fit


There are several distributions which are implemented in this library,
all of which extend the abstract class :class:`powerlaw.Distribution`.


.. autosummary::

    powerlaw.Distribution

    powerlaw.Power_Law
    powerlaw.Truncated_Power_Law 
    powerlaw.Exponential
    powerlaw.Stretched_Exponential
    powerlaw.Lognormal
    powerlaw.Lognormal_Positive

Besides these main classes, there are some helper functions defined in other
modules:

.. autosummary::

    powerlaw.data
    powerlaw.utils
    powerlaw.plotting
    powerlaw.statistics


