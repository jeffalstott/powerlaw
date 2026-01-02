Truncated Power Law Derivations
---------------------

This page covers the analytics and derivations relevant to the
exponentially-truncated power law distribution.

Probability distribution function (PDF)
=======================================

The PDF for a truncated power law distribution has the form:

.. math::

    p(x) = C x^{-\alpha} e^{-\lambda x}

Defined over the domain :math:`[x_{min}, x_{max}]`, and for :math:`\alpha \ne 1`,
this has normalization constant:

.. math::

   C = \left[ \int_{x_{min}}^{x_{max}} x^{-\alpha} e^{-\lambda x} dx \right]^{-1}

This is nearly the definition of the incomplete gamma function, :math:`\Gamma_s(z)` (see TODO):

.. math::

   = - \lambda^{1 - \alpha} \left[ \Gamma_{1 - \alpha}(\lambda x_{max}) - \Gamma_{1 - \alpha}(\lambda x_{min}) \right]^{-1}

Unlike the regular power law, there is (in principle) no issue with having :math:`\alpha < 1`
since the normalization will certainly be finite due to the exponential cutoff. That
being said, in practice there are some difficulties evaluating this gamma
function (and it's inverse) for negative values of the parameter :math:`s`
(ie. :math:`1 - \alpha`).

TODO: Check above, since it would be positive for alpha less than 1

The gamma function goes to zero as its argument goes to infinity, so for
:math:`x_{max} \rightarrow \infty`, we have:

.. math::

   C = \frac{\lambda^{1 - \alpha}}{\Gamma_{1 - \alpha}(\lambda x_{min})}

Cumulative distribution function (CDF)
=======================================

The CDF for a truncated power law distribution is:

.. math::
    
    c(x) = C \int_{x_{min}}^{x} {x'}^{-\alpha} e^{-\lambda x'} dx'

.. math::

    = - \frac{C}{\lambda^{1 - \alpha}} \left[ \Gamma_{1 - \alpha}(\lambda x) - \Gamma_{1 - \alpha}(\lambda x_{min}) \right]

.. math::

    = \frac{\Gamma_{1 - \alpha}(\lambda x) - \Gamma_{1 - \alpha}(\lambda x_{min}) }{ \Gamma_{1 - \alpha}(\lambda x_{max}) - \Gamma_{1 - \alpha}(\lambda x_{min}) }

For a semi-infinite domain (:math:`x_{max} \rightarrow \infty`), this simplifies
to:

.. math::

    c(x) = 1 - \frac{ \Gamma_{1 - \alpha}(\lambda x) }{ \Gamma_{1 - \alpha}(\lambda x_{min}) }

Inverse CDF
===========

In order to generate random numbers, we use inverse transform sampling,
which involves inverting the CDF. For an arbitrary value of :math:`x_{max}`,
this is:

.. math::

    r = \frac{\Gamma_{1 - \alpha}(\lambda c^\dagger(r)) - \Gamma_{1 - \alpha}(\lambda x_{min}) }{ \Gamma_{1 - \alpha}(\lambda x_{max}) - \Gamma_{1 - \alpha}(\lambda x_{min}) }

.. math::

    r \left( \Gamma_{1 - \alpha}(\lambda x_{max}) - \Gamma_{1 - \alpha}(\lambda x_{min}) \right) = \Gamma_{1 - \alpha}(\lambda c^\dagger(r)) - \Gamma_{1 - \alpha}(\lambda x_{min})

.. math::

    r \Gamma_{1 - \alpha}(\lambda x_{max}) + (1 - r) \Gamma_{1 - \alpha}(\lambda x_{min}) = \Gamma_{1 - \alpha}(\lambda c^\dagger(r))

.. math::

    c^\dagger(r) = \frac{1}{\lambda} \Gamma_{1 - \alpha}^\dagger \left( r \Gamma_{1 - \alpha}(\lambda x_{max}) + (1 - r) \Gamma_{1 - \alpha}(\lambda x_{min}) \right)

For a semi-infinite domain (:math:`x_{max} \rightarrow \infty`), this simplifies
to:

.. math::

    c^\dagger(r) = \frac{1}{\lambda} \Gamma_{1 - \alpha}^\dagger \left( (1 - r) \Gamma_{1 - \alpha}(\lambda x_{min}) \right)

Unfortunately, implementing this function in Python is currently not possible
since there doesn't exist an implementation of the inverse incomplete
gamma function that can handle negative parameter values.

The two most prominent options would be scipy's ``scipy.special.gammainc`` and
``scipy.special.gammaincinv``, or mpmath's ``mpmath.gammainc``. There is
discussion of adding support for negative parameters for this gamma function
in scipy (see `here <https://github.com/scipy/scipy/issues/21498>`_), though we
would still need the inverse as well.

TODO: Talk about temporary solution
