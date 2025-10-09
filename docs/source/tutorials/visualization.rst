Visualization
=============


The ``powerlaw`` package supports easy plotting of the probability density
function (PDF), the cumulative distribution function (CDF; :math:`p(X<x)` ) and
the complementary cumulative distribution function (CCDF; :math:`p(X\geq x)`,
also known as the survival function).


Even before trying to fit your data using the :meth:`powerlaw.Fit` class, you can
compute and visualize your data using the following functions:

.. autosummary::

    powerlaw.pdf
    powerlaw.cdf
    powerlaw.ccdf

.. autosummary::

    powerlaw.plot_pdf
    powerlaw.plot_cdf
    powerlaw.plot_ccdf

So to plot the PDF and CDF of a dataset we might do:

.. code-block::

    data = np.array([1.7, 3.2 ...])

    powerlaw.plot_pdf(data)
    powerlaw.plot_cdf(data)

These plotting commands accept matplotlib keyword arguments.
