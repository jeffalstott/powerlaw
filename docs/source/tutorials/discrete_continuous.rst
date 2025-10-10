Continuous and discrete data
============================

Datasets are treated as continuous by default, and thus fit to continuous
forms of distributions. Many data are discrete, however, and cannot be
accurately fitted with continuous distributions [1]. Discrete (integer)
distributions, with proper normalizing, can be dictated at initialization:

.. code-block::

    # This is a continuous dataset
    data = powerlaw.load_test_dataset('quakes')
    fit = powerlaw.Fit(data)

    # This is a discrete dataset, so we should specify that
    data = powerlaw.load_test_dataset('words')
    fit = powerlaw.Fit(data, discrete=True)

Discrete forms of probability distributions are frequently more difficult
to calculate than continuous forms, and so certain computations may be
slower. However, there are approximations that can be used in place of
some of these calculations.

The maximum likelihood fit to a continuous power law for a given
:math:`x_{min}` can be calculated analytically for certain values of :math:`\alpha`
(see :meth:`powerlaw.Power_Law.generate_initial_parameters`). This means
that the optimal :math:`x_{min}` and resulting fitted parameters can be
computed quickly. This is not so for the discrete case. The maximum likelihood
fit for a discrete power law is found by numerical optimization,
and thus performing this computation for every possible :math:`x_{min}` can take
time. To circumvent this issue, ``powerlaw`` can use an analytic estimate
of :math:`\alpha` which can "give results accurate to about 1\% or better
provided :math:`x_{min} \ge 6`" when not using an :math:`x_{max}` [1].
This option, ``estimate_discrete``, is ``True`` by default whenever these
conditions are satisfied:


References
----------

[1] Clauset A, Shalizi CR, Newman MEJ (2009) Power-law distributions in
empirical data. SIAM Review 51.
