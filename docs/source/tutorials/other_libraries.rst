Differences with other libraries
==============================

There are many different fitting libraries for Python, and it might not
be immediately obvious the difference between ``powerlaw`` and others like
`lmfit <https://lmfit.github.io/lmfit-py/>`_ or even just `scipy <https://scipy.org/>`_.

In these latter libraries, a fit is performed by defining some cost function
:math:`C(\vec p)` with parameters :math:`\vec p`, and then finding the
particular parameter values that minimize that cost function. Often for
fitting a particular function :math:`f(x, \vec p)` to some data, you will define this
cost as some difference between the fitted values and the true values:

.. code-block::

    # Some function, eg. a power law
    def fit_function(x, p):
        return p[0] * x**(p[1])

    # Some data values
    y = ...

    # Some cost, eg. MSE
    def cost_function(p):
        return np.sqrt(np.sum((y - fit_function(x, p))**2))

    # Now we minimize this with scipy, lmfit, or some other library
    minimize(cost, p0, ...)

As mentioned in `the tutorial on loading data <loading_data.html>`_, 
our input data is a collection of samples that we expect arise from a
particular distribution. Since they are samples, we don't actually want
to fit a function to the specific samples we have, but rather to the
probabilities of finding these specific samples. This means that compared to
the above example, we have an extra function used within our cost function:
the probability distribution (PDF).

.. code-block::

    # This compares the original data to the fit function
    def cost_function(p):
        return distance_function(y - fit_function(p))

    # This compares the PDF of the original data to the fit function
    # (which is a PDF).
    def cost_function(p):
        return distance_function(pdf(y) - fit_function(p))


While the usual workflow of ``powerlaw`` doesn't look like this, it is only
because ``fit_function``, ``cost_function``, etc. are all defined within
the package, and these are all automatically set by choosing a particular
distribution, for example ``power_law``. Let's examine what some of these
fit functions and cost functions look like for the power law. This fit
function is indeed the PDF for a power law:

.. code-block::

    def Power_Law(Distribution):
        
        ...

        def _pdf_base_function(self, x):
            ...
            return x**(-self.alpha)

So far this isn't fundamentally different from the other libraries, but this
difference comes in now. Note how there's only a single parameter -- the exponent --- instead of
the very first case above where we had two parameters --- the exponent and the
normalization factor. This library is more specialized than a general
fitting library like ``lmfit`` in that it assumes everything being fit
is a proper probability distribution, which must have very specific
mathematical properties. As such, we can compute the normalization constant
from these properties, and don't need to (and more so, we don't `want` to) fit
this constant with an arbitrary value.

So unlike other fitting libraries, a fit function in ``powerlaw`` (a
distribution) contains much more than just a single function, but must
have a PDF (eg. :meth:`Power_Law._pdf_base_function`), a cumulative distribution
function (CDF, eg. :meth:`Power_Law._cdf_base_function`), an expression
for computing the normalization constant (eg. :meth:`Power_Law._pdf_continuous_normalizer`),
and potentially several other functions.

So what do you gain for putting in all of this extra work to implement
(and maybe more difficult than that, derive) these functions? A few things:

1. In several cases, by making the assumption that your fit function is a
probability distribution, you can derive an analytical expression for the
maximum likelihood estimate, or fit, of your data. This means no numerical
fitting (think ``scipy.optimize.minimize`` type of function) is required at
all, and you can immediately calculate the best parameter values in :math:`\mathcal{O}(1)`.

2. Even in cases where such an analytical expression isn't available, the
number of parameters you have to fit is lower, since certain values will
be constrained by the assumption that your function is a probability
distribution.

3. Having analytical representations of distributions allows for comparisons
between different types of fits beyond simple residual and error analysis.
This includes likelihood measures or distribution distances such as
Kolmogorov-Smirnov distance. When analyzing real sampled data, it is usually
not that important or informative to show that your fitting error is under some
particular (usually arbitrary) threshold. This is especially true when you
consider that you can always fit you data with an arbitrary number of parameters
to drive that fitting error to zero. Instead it is more convincing --- to
us at least --- to show which physically-plausible distribution fits your
data the best in comparison to other candidate distributions. For a more
in-depth discussion, see the `Comparing distributions <comparing_distributions.html>`_
tutorial.

Most of the time this extra work required to implement a fit function is
not relevant anyway, since many of the most common distributions measured in nature
are already implemented in ``powerlaw``, and thus require almost no work from the end user.
