Identifying the scaling range
=============================

The first step of fitting a power law is to determine what portion of the
data to fit. A heavy-tailed distribution's interesting feature is the tail
and its properties, so if the initial, small values of the data do not
follow a power law distribution the user may opt to disregard them. On the
other hand, the larger values of the distribution may not be experimentally
accessible, so you might want to restrict the fitting to below a certain point.


Lower limits (:math:`x_{min}`)
------------------------------

The easiest case is when the value of :math:`x_{min}` is known, whether
from measurement limitations, theoretical models, or whatever else.
When creating a ``Fit`` object, this can be directly specified:

.. code-block::

    data = ...
    xmin = 1 # Or some other value

    fit = powerlaw.Fit(data, xmin=xmin)

If you don't have a particular value for :math:`x_{min}` in mind, ``powerlaw``
implements the methods described in Clauset et al. (2009) to find this value.

This is done by creating a power law fit starting from each unique value in
the dataset, and then selecting the one that results in the minimal distance
between the data and the fit. This distance can be calculated according to
different schemes, but the most common are:

* Kolmogorov-Smirnov distance, :math:`D`
* Kuiper distance, :math:`V`
* Anderson-Darling distance, :math:`A^2`

This optimization is done in :meth:`Fit.find_xmin`, which is automatically
called whenever you create a ``Fit`` object without specifying a value for
``xmin``.

.. code-block::

    # This will use find_xmin() to get the best value for xmin.
    fit = powerlaw.Fit(data)

By default, this will use the Kolmogorov-Smirnov distance as the evaluation
metric, but this can be changed with the ``xmin_distance`` keyword argument
to the ``Fit`` class.

Since this often involves fitting the data quite a lot, it is usually the
most computationally expensive function call in the library. If you'd like
to see how long it is taking with a progress bar, you can use ``verbose=2``
when creating your ``Fit`` instance.

.. code-block::

    # This will use find_xmin() to get the best value for xmin and
    # show a progress bar.
    fit = powerlaw.Fit(data, verbose=2)

If you'd like to restrict the range that the algorithm attempts to find a
value for :math:`x_{min}`, you can pass a tuple specifying the bounds instead
of a single value:

.. code-block::

    # This will use find_xmin() to get the best value for xmin in the
    # fixed range 50 to 100.
    fit = powerlaw.Fit(data, xmin=(50, 100))

If you later want to see whether a fit was given an ``xmin`` value or if it
was fit, you can examine the property ``fixed_xmin`` of the ``Fit`` class.

Upper limits (:math:`x_{max}`)
------------------------------

In some domains there may also be an expectation that the distribution will
have a precise upper bound, :math:`x_{max}`. An upper limit could be due a
theoretical limit beyond which the data simply cannot go (eg. in
astrophysics, a distribution of speeds could have an upper bound at the
speed of light). An upper limit could also be due to finite-size scaling,
in which the observed data comes from a small subsection of a larger system.
The finite size of the observation window would mean that individual data
points could be no larger than the window, :math:`x_{max}`, though the greater
system would have larger, unobserved data (eg. in neuroscience, recording
from a patch of cortex vs the whole brain). Finite-size effects can be tested
by experimentally varying the size of the observation window (and
:math:`x_{max}`) and determining if the data still follows a power law with
the new :math:`x_{max}` [1, 2]. The presence of an
upper bound relies on the nature of the data and the context in which it
was collected, and so can only be dictated by the user.

If one is given, any data above :math:`x_{max}` is ignored for fitting. 

.. code-block::

    # Any data that is larger than 100 will be ignored in fitting
    xmax = 100
    fit = powerlaw.Fit(data, xmax=xmax)

Effects of upper and lower limits
---------------------------------


For calculating or plotting CDFs, CCDFs, and PDFs, by default ``Fit`` objects
only use data above :math:`x_{min}` and below :math:`x_{max}` (if present).
The ``Fit`` object's plotting commands can plot all the data originally given
to it with the keyword ``original_data=True``.

.. code-block::

    xmin = 10
    xmax = 1e4

    fit = powerlaw.Fit(data, xmin=xmin, xmax=xmax)

    fit.plot_pdf(label='Cropped PDF')
    fit.plot_pdf(original_data=True, label='Full PDF')

    ...

.. figure:: ../images/fires_pdf_cropped.png

    PDF for the ``fires`` dataset within the specified scaling range 
    ``[1e1, 1e4]`` and over the whole data range.

The constituent ``Distribution`` objects are only defined within the range
of :math:`x_{min}` and :math:`x_{max}`, but can be plotted in any subset of
that range by passing specific data with the keyword ``data``. 
 
When using an :math:`x_{max}`, a power law's CDF and CCDF do not appear in
a straight line on a log-log plot, but bend down as the :math:`x_{max}` is
approached. The PDF, in contrast, appears straight all way to :math:`x_{max}`.
Because of this, PDFs are preferable when visualizing data with an :math:`x_{max}`,
so as to not obscure the scaling.

.. subfigure:: AB

    .. image:: ../images/words_pdf_cropped.png

    .. image:: ../images/words_ccdf_cropped.png

    PDF (left) and CCDF (right) for the ``words`` dataset within the
    specified scaling range ``[1e1, 1e4]`` and over the whole data range.
    The curvature of the CCDF changes when cropped, but that of the PDF remains
    the same.

References
----------

[1] Beggs JM, Plenz D (2003) Neuronal Avalanches in Neocortical Circuits.
    The Journal of Neuroscience 23: 11167–11177.

[2] Shriki O, Alstott J, Carver F, Holroyd T, Henson R, et al. (2013)
    Neuronal Avalanches in the Resting MEG of the Human Brain. Journal of
    Neuroscience 33: 7079–7090
