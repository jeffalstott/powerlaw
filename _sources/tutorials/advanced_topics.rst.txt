Advanced Topics
===============

This page covers many eclectic topics that don't fit elsewhere, and generally
are only relevant for advanced users.

No possible fits
----------------

When fitting a distribution to data, there may be no valid fits. This would
most typically arise from overly strict fitting requirements, but it could
also happen because the data truly doesn't fit the specified form. This
latter case is easy enough to test (just look at the PDF of the data), but
the former case can be a little harder to fix.

First, in order to diagnose this issue, you should look at the ``noise_flag``
property of either the ``Fit`` object (signifying that every value of ``xmin``
failed) or for a specific distribution object.

.. code-block::

    data = powerlaw.load_test_dataset('blackouts')

    # pl.noiseflag will be True since the powerlaw exponent of the
    # blackouts dataset should be around 2.4, outside of the prescribed
    # range.
    pl = powerlaw.Power_Law(data, parameter_ranges={"alpha": [1, 2]})
    pl.noise_flag

    # fit.noiseflag will be True since the prescribed sigma_threshold
    # is too low
    # range.
    fit = powerlaw.Fit(data, sigma_threshold=1e-12)
    pl.noise_flag

One way to fix this is by adjusting your parameter ranges, threshold, constraints,
etc. If you worry that the fitting landscape is too noisy, you can try
providing an explicit initial condition:

.. code-block::
    
    fit = powerlaw.Fit(data, initial_parameters={"alpha": 2.4})

If you're seeing a warning message about something, fixing that may help
to get better fitting results; see the tutorial about `dealing with warnings
<warnings.html>`_ for more information.


Multiple possible fits
----------------------

Changes in :math:`x_{min}` with different parameter requirements illustrate
that there may be more than one fit to consider. Assuming there is no
:math:`x_{max}`, the optimal :math:`x_{min}` is selected by finding the
:math:`x_{min}` value with the lowest distance (see ``xmin_distance`` keyword
for :class:`powerlaw.Fit`) between the data and the fit for that :math:`x_{min}`
value. If there is only one local minimum across all :math:`x_{min}` values,
this is philosophically simple.

If, however, there are multiple local minima in this distance metric across
:math:`x_{min}` with similar minimum values, it may be worth noting and
considering these alternative fits. For this purpose, :class:`powerlaw.Fit`
retains information on all the possible :math:`x_{min}`, along with the
parameters of the subsequent fit, in the ``xmin_fitting_results`` property.

As an example, let's examine the fitting results for the ``blackouts`` data:

.. code-block::

    data = powerlaw.load_test_dataset('blackouts')

    fit = powerlaw.Fit(data / 1e3)

    # Now examine the xmin fitting results
    plt.plot(fit.xmin_fitting_results["xmin"], fit.xmin_fitting_results["distance"])
   
.. figure:: ../images/blackouts_multiple_fits.png

    The Kolmogorov-Smirnov distance and :math:`\alpha` values for the resulting fit for every possible
    value of :math:`x_{min}` for the ``blackouts`` data set.

The first minima is at :math:`x_{min} = 50`, and has a distance value of .1
and an :math:`\alpha` value of 1.78. The second, more optimal, fit is at
:math:`x_{min} = 230`, with a distance of .06 and :math:`\alpha = 2.27`.
The reality of which of these is a better fit to the data will depend
on exactly what type of data you have, and isn't always obvious if there
are fits with similar distance values, as above.
For example, if you expect that the power law exponent should be restricted
to a specific range, say greater than 2, the second minima would be the
most correct choice. If you have expectations about such values, it is
a good idea to try and encode this in the parameter ranges or constraints
of fitting.

.. code-block::

    data = powerlaw.load_test_dataset('blackouts')

    fit = powerlaw.Fit(data / 1e3, parameter_ranges={"alpha": [2, 3]})

    ...

.. figure:: ../images/blackouts_multiple_fits_bounded.png

    The Kolmogorov-Smirnov distance and :math:`\alpha` values for the resulting fit for every possible
    value of :math:`x_{min}` for the ``blackouts`` data set while constraining
    :math:`\alpha` to be in the range :math:`[2, 3]`.

For more information, see the tutorial about `restricting fits with ranges and
constraints <ranges_and_constraints.html>`_.
 

Maximum likelihood and independence assumptions
-----------------------------------------------

A fundamental assumption of the maximum likelihood method used for fitting,
as well as the loglikelihood ratio test for comparing the goodness of fit of
different distributions, is that individual data points are independent [1].
In some datasets, correlations between observations may be known or expected.
For example, in a geographic dataset of the elevations of peaks, of the
observation of a mountain of height :math:`X` could be correlated with the
observation of foothills nearby of height :math:`X/10`. Large correlations
can potentially greatly alter the quality of the maximum likelihood fit.
Theoretically, such correlations may  be incorporated into the likelihood
calculations, but doing so would greatly increase the computational
requirements for fitting.
 
Depending on the nature of the correlation, some datasets can be "decorrelated"
by selectively omitting portions of the data [2]. Using the foothills
example, the correlated foothills may be known to occurr within 10km of a
mountain, and beyond 10km the correlations drops to 0. Requiring a minimum
distance of 10km between observations of peaks, and omitting any additional
observations within that distance, would decorrelate the dataset. 
 
An alternative to maximum likelihood estimation is minimum distance estimation,
which fits the theoretical distribution to the data by minimizing the
Kolmogorov-Smirnov (or some other) distance between the data and the fit.
This can be specified by passing ``fit_method='ks'`` to the ``Fit``; the
particular distance, whether Kolmogorov-Smirnov, Anderson-Darling, etc. is
controlled with the ``xmin_distance`` keyword. However, the use of this
option will not solve the problem of correlated data points for the
loglikelihood ratio tests used in :meth:`powerlaw.Fit.distribution_compare`.


Power Laws vs. Lognormals and powerlaw's 'lognormal_positive' option
--------------------------------------------------------------------

When fitting a power law to a data set, among the possible alternatives, one
should take particular care to compare the goodness of fit to that of a
`lognormal distribution <https://en.wikipedia.org/wiki/Lognormal_distribution>`__.
This is because lognormal distributions are another heavy-tailed distribution,
but they can be generated by a very simple process: multiplying random positive
variables together. The lognormal is thus much like the normal distribution,
which can be created by adding random variables together; in fact, the log of
a lognormal distribution is a normal distribution (hence the name), and the
exponential of a normal distribution is the lognormal (which maybe would be
better called an expnormal). In contrast, creating a power law generally
requires fancy or exotic generative mechanisms (this is probably why you're
looking for a power law to begin with; they're sexy). So, even though the
power law has only one parameter (``alpha``: the exponent) and the lognormal
has two (``mu``: the mean of the random variables in the underlying normal
and ``sigma``: the standard deviation of the underlying normal distribution),
we typically consider the lognormal to be a simpler explanation for observed
data, as long as the distribution fits the data just as well. For most data
sets, a power law is actually a worse fit than a lognormal distribution, or
perhaps equally good, but rarely better. This fact was one of the central
empirical results of the paper `Clauset et al. 2007 <http://arxiv.org/abs/0706.1062>`__,
which developed the statistical methods that ``powerlaw`` implements.

However, for many data sets, the superior lognormal fit is only possible if
one allows the fitted parameter ``mu`` to go negative. Whether or not this
is sensible depends on your theory of what's generating the data. If the
data is thought to be generated by multiplying random positive variables,
``mu`` is just the log of the distribution's median; a negative ``mu`` just
indicates those variables' products are typically below 1. However, if the
data is thought to be generated by exponentiating a normal distribution, then
``mu`` is interpreted as the median of the underlying normal data. In that case,
the normal data is likely generated by summing random variables (positive
and negative), and ``mu`` is those sums' median (and mean). A negative
``mu``, then, indicates that the random variables are typically negative.
For some physical systems, this is perfectly possible. For the data you're
studying, though, it may be a weird assumption. For starters, all of the data
points you're fitting to are positive by definition, since power laws must
have positive values (indeed, ``powerlaw`` throws out zeros and negative
values). Why would those data be generated by a process that sums and
exponentiates *negative* variables?

If you think that your physical system could be modeled by summing and
exponentiating random variables, but you think that those random variables
should be positive, one possible hacks is ``powerlaw``'s ``lognormal_positive``.
This is just a regular lognormal distribution, except ``mu`` must be positive,
essentially equivalent to just setting the parameter range for ``mu`` to be
``[0, None]``.
Note that this does not force the underlying normal distribution to be the
sum of only positive variables; it only forces the sums' *average* to be
positive, but it's a start. You can compare a power law to this distribution
in the normal way:

.. code-block:: 

    R, p = results.distribution_compare('power_law', 'lognormal_positive')

You may find that a lognormal where ``mu`` must be positive gives a much
worse fit to your data, and that leaves the power law looking like the best
explanation of the data. Before concluding that the data is in fact power
law distributed, consider carefully whether a more likely explanation is that
the data was generated by multiplying positive random variables, or even by
summing and exponentiating random variables; either one would allow for a
lognormal with an intelligible negative value of ``mu``.


Numerical fitting arguments
---------------------------

When numerical fitting is necessary (see tutorials on `other libraries <other_libraries.html>`_
or `continuous and discrete <discrete_continuous.html>`_), this library
uses ``scipy.optimize.minimize``. This function is incredibly deep and
decisions about parameter values could be discussed ad nauseam; we give a
brief overview of how this function is used here, but see the source code
--- particularly the :meth:`powerlaw.Distribution.fit` function --- for
more information.

As of v1.6.0, there are two options for fitting method used: Nelder-Mead
when there are no constraints, and COBYLA when there are constraints.
Both of these have been roughly tested across a range of distributions
(though primarily power laws) and they seem to work well enough. Otherwise,
there is no particular reason why these optimization methods are used.


References
----------

[1] Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law
distributions in empirical data. SIAM Review, 51(4), 661–703.
https://doi.org/10.1137/070710111

[2] Klaus, A., Yu, S., & Plenz, D. (2011). Statistical Analyses Support
Power Law Distributions Found in Neuronal Avalanches. PLOS ONE, 6(5),
e19779. https://doi.org/10.1371/journal.pone.0019779

