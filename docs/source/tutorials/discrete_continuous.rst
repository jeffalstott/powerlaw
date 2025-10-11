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

Normalization approximations
----------------------------

In general, the discrete forms of some distributions are not analytically
defined (eg. lognormal and stretched exponential). This often means that there
is no good way to compute the normalization constant if you assume the data
takes on only integer values. In such cases, there are two available
approximations of the discrete form.

The easiest (but computationally somewhat expensive) approximation is 
to simply calculate the normalization constant by summing the (continuous) probability
distribution evaluated at every integer value over the entire range that it is defined.
The lower limit of this range will be ``xmin``, and the upper limit will be
either ``xmax`` or the maximum value in the sampled data. This method is
called ``'sum'``:

.. code-block::

    fit = powerlaw.Fit(data, discrete=True, discrete_normalization='sum')

The second approximation method is discretization by rounding, in which the
continuous distribution is summed in a small range around each point. In
this case, the probability mass at any value of :math:`x` is equal to the
sum of the continuous probability in the range :math:`[x - 0.5, x + 0.5]`
Because of its speed, this rounding method is the default. This method is
called ``'round'``:

.. code-block::

    fit = powerlaw.Fit(data, discrete=True, discrete_normalization='round')


Distribution-specific approximations
------------------------------------

When they are available, discrete forms of probability distributions are
frequently more difficult to calculate than continuous forms, and so certain
computations may be slower. However, there are approximations that can be
used in place of some of these calculations.

The maximum likelihood fit to a continuous power law for a given
:math:`x_{min}` can be calculated analytically for certain values of :math:`\alpha`
(see :class:`powerlaw.Power_Law`). This means
that the optimal :math:`x_{min}` and resulting fitted parameters can be
computed quickly. This is not so for the discrete case. The maximum likelihood
fit for a discrete power law is found by numerical optimization,
and thus performing this computation for every possible :math:`x_{min}` can take
time. To circumvent this issue, ``powerlaw`` can use an analytic estimate
of :math:`\alpha` which can "give results accurate to about 1\% or better
provided :math:`x_{min} \ge 6`" when not using an :math:`x_{max}` [1].
This option, ``estimate_discrete``, is ``True`` by default whenever these
conditions are satisfied, though can be explicitly set to ``True`` or ``False``
as desired:

.. code-block::

    # If the conditions described above are satisfied, this
    # will use the discrete estimation technique.
    fit = powerlaw.Fit(data, discrete=True)

    # This will require the use of the estimation technique,
    # regardless of the properties of data.
    # Be careful when doing this, as you may get very incorrect
    # fits if you data violates the conditions!
    fit = powerlaw.Fit(data, discrete=True, estimate_discrete=True)


Generation of simulated data from a theoretical distribution has similar
considerations for speed and accuracy. There is no rapid, exact calculation
method for random data from discrete power law distributions. Generated data
can be calculated with a fast approximation or with an exact search
algorithm that can run several times slower [1]. The desired option is again
selected by using the ``estimate_discrete`` keyword, which can be specified
at the creation of the ``Fit`` object, or at each call of the function
:meth:`Distribution.generate_random`.

The fast estimation of random data has an error that scales with the
:math:`x_{min}`. When :math:`x_{min} = 1` the error is over 8\%, but at
:math:`x_{min} = 5` the error is less than 1\% and at :math:`x_{min} = 10`
less than .2\% [1]. Thus, for distributions with small values of :math:`x_{min}`
the exact calculation is likely preferred.  

.. subfigure:: AB

    .. image:: ../images/words_generation_1_xmin.png

    .. image:: ../images/words_generation_10_xmin.png

    PDF for the ``words`` dataset compared to random data generated from
    this distribution using :meth:`Distribution.generate_random` for
    :math:`x_{min}` values of 1 (left) and 10 (right). The left plot
    shows a much larger difference between the estimated random data and
    the true data when compared to the right plot.

References
----------

[1] Clauset A, Shalizi CR, Newman MEJ (2009) Power-law distributions in
empirical data. SIAM Review 51.
