Stretched Exponential Derivations
---------------------

This page covers the analytics and derivations relevant to the stretched exponential
distribution.

Probability distribution function (PDF)
=======================================

The PDF for a stretched exponential distribution has the form:

.. math::

    p(x) = C x^{\beta - 1} e^{-(\lambda x)^\beta}

Defined over the domain :math:`[x_{min}, x_{max}]`, this has normalization constant:

.. math::

   C = \left[ \int_{x_{min}}^{x_{max}} x^{\beta - 1} e^{-(\lambda x)^\beta} dx \right]^{-1}
   =  - \beta \lambda^\beta \left[ e^{-(\lambda x_{max})^\beta} - e^{-(\lambda x_{min})^\beta} \right]^{-1}

For a semi-infinite domain (:math:`x_{max} \rightarrow \infty`) and typical
values of :math:`\beta \in (0, 1)`, this simplifies to:

.. math::

    C = \beta \lambda^\beta e^{(\lambda x_{min})^\beta}

As with regular exponentials, this can cause overflow and underflow issues if we store this
normalization constant as a separate value from the unnormalized PDF values.

Cumulative distribution function (CDF)
=======================================

The CDF for a stretched exponential distribution is:

.. math::
    
    c(x) = C \int_{x_{min}}^{x} {x'}^{\beta - 1} e^{-(\lambda x')^\beta} dx'

.. math::

    = - \frac{C}{\beta \lambda^\beta} \left[ e^{-(\lambda x)^\beta} - e^{-(\lambda x_{min})^\beta} \right]

.. math::

    =  \frac{ e^{-(\lambda x)^\beta} - e^{-(\lambda x_{min})^\beta} }{ e^{-(\lambda x_{max})^\beta} - e^{-(\lambda x_{min})^\beta} }

For a semi-infinite domain (:math:`x_{max} \rightarrow \infty`), this simplifies
to:

.. math::

    c(x) = 1 - e^{-\lambda^\beta (x^\beta - x_{min}^\beta)}

Inverse CDF
===========

In order to generate random numbers, we use inverse transform sampling,
which involves inverting the CDF. For an arbitrary value of :math:`x_{max}`,
this is:

.. math::

    r =  \frac{ e^{-(\lambda c^\dagger(r))^\beta} - e^{-(\lambda x_{min})^\beta} }{ e^{-(\lambda x_{max})^\beta} - e^{-(\lambda x_{min})^\beta} }

.. math::

    r \left( e^{-(\lambda x_{max})^\beta} - e^{-(\lambda x_{min})^\beta} \right) =  e^{-(\lambda c^\dagger(r))^\beta} - e^{-(\lambda x_{min})^\beta}

.. math::

    r e^{-(\lambda x_{max})^\beta} + (1 - r) e^{-(\lambda x_{min})^\beta} = e^{-(\lambda c^{\dagger}(r))^\beta}

.. math::

    c^\dagger(r) = \frac{1}{\lambda} \left[ -\log \left( r e^{-(\lambda x_{max})^\beta} + (1 - r) e^{-(\lambda x_{min})^\beta} \right) \right]^{\frac{1}{\beta}}

For a semi-infinite domain (:math:`x_{max} \rightarrow \infty`), this simplifies
to:

.. math::

    c^\dagger(r) = \frac{1}{\lambda} \left[ - \log \left( (1 - r) e^{-(\lambda x_{min})^\beta} \right) \right]^{\frac{1}{\beta}}

.. math::

    c^\dagger(r) = \frac{1}{\lambda} \left[ (\lambda x_{min})^\beta - \log (1 - r) \right]^{\frac{1}{\beta}}

In order to generate random numbers that follow a stretched exponential distribution,
we first generate uniform random numbers in the range :math:`[0, 1)`, and
then pass these values through the inverse CDF. The random numbers produced
by this will be in the range :math:`[x_{min}, \infty)` (if we use the simplified
expression above) or :math:`[x_{min}, x_{max})` (if we use the full expression
with :math:`x_{max}`).

Another way to generate bounded random numbers while still using the
simplified expression is to restrict the range of uniform random numbers to
:math:`[0, c(a))`, where now the resulting samples will be in the range
:math:`[x_{min}, a)`. This latter method is a little more convenient, since
it means we can use the same inverse CDF to generate both bounded and unbounded
random numbers.
