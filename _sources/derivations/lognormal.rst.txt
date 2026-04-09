Lognormal Derivations
---------------------

This page covers the analytics and derivations relevant to the lognormal
distribution.

Probability distribution function (PDF)
=======================================

The PDF for a lognormal distribution has the form:

.. math::

    p(x) = C \frac{1}{x} e^{-\frac{(\log{x} - \mu)^2}{2\sigma^2}}

Defined over the domain :math:`[x_{min}, x_{max}]`, this has normalization constant:

.. math::

   C = \left[ \int_{x_{min}}^{x_{max}} \frac{1}{x} e^{-\frac{(\log{x} - \mu)^2}{2\sigma^2}} dx \right]^{-1}

After simplifying slightly, this is an error function integral (you could also just use a software like Mathematica):

.. math::

   C = \left[ \int_{x_{min}}^{x_{max}} \frac{1}{x} e^{-\frac{1}{2\sigma^2} \left( \log^2 x - 2 \mu \log x + \mu^2 \right) } dx \right]^{-1}

.. math:: 

   = e^{\mu^2 / 2 \sigma^2} \left[ \int_{x_{min}}^{x_{max}} \frac{1}{x} e^{-\frac{1}{2\sigma^2} \left( \log^2 x - 2 \mu \log x \right) } dx \right]^{-1}

.. math:: 

   = e^{\mu^2 / 2 \sigma^2} \left[ - \left. e^{\mu^2 / 2 \sigma^2} \sqrt{\frac{\pi \sigma^2}{2}} \text{erf} \left( \frac{\mu - \log x}{\sqrt{2 \sigma^2}} \right)  \right|_{x_{min}}^{x_{max}} \right]^{-1}

.. math:: 

   = - \sqrt{\frac{2}{\pi \sigma^2}} \left[ \text{erf} \left( \frac{\mu - \log x_{max}}{\sqrt{2 \sigma^2}} \right) - \text{erf} \left( \frac{\mu - \log x_{min}}{\sqrt{2 \sigma^2}} \right) \right]^{-1}

For a semi-infinite domain (:math:`x_{max} \rightarrow \infty`), this simplifies
to:

.. math:: 

   C = - \sqrt{\frac{2}{\pi \sigma^2}} \left[ -1 - \text{erf} \left( \frac{\mu - \log x_{min}}{\sqrt{2 \sigma^2}} \right) \right]^{-1}

.. math:: 

    = \sqrt{\frac{2}{\pi \sigma^2}} \left[ \text{erfc} \left( \frac{\log x_{min} - \mu}{\sqrt{2 \sigma^2}} \right) \right]^{-1}

It is also somewhat common to set :math:`x_{min} = 0` (see, for example,
the `Wikipedia page <https://en.wikipedia.org/wiki/Log-normal_distribution>`_),
which allows us to simplify further:

.. math:: 

    C = \sqrt{\frac{2}{\pi \sigma^2}} \left[ 2 \right]^{-1}

.. math:: 

    C = \sqrt{\frac{1}{2 \pi \sigma^2}}

Cumulative distribution function (CDF)
=======================================

The CDF for a lognormal distribution is:

.. math::
    
   c(x) = C \int_{x_{min}}^{x} \frac{1}{x'} e^{-\frac{(\log{x'} - \mu)^2}{2\sigma^2}} dx'

This is almost exactly the same integral as above, so we can just skip to
the result:

.. math::

   c(x) = - C \sqrt{\frac{\pi \sigma^2}{2}} \left[ \text{erf} \left( \frac{\mu - \log x}{\sqrt{2 \sigma^2}} \right) - \text{erf} \left( \frac{\mu - \log x_{min}}{\sqrt{2 \sigma^2}} \right) \right]

.. math::

   = \frac{ \text{erf} \left( \frac{\mu - \log x}{\sqrt{2 \sigma^2}} \right) - \text{erf} \left( \frac{\mu - \log x_{min}}{\sqrt{2 \sigma^2}} \right) }{ \text{erf} \left( \frac{\mu - \log x_{max}}{\sqrt{2 \sigma^2}} \right) - \text{erf} \left( \frac{\mu - \log x_{min}}{\sqrt{2 \sigma^2}} \right) }

For a semi-infinite domain (:math:`x_{max} \rightarrow \infty`), this simplifies
to (using :math:`\lim_{y \rightarrow \pm \infty} \text{erf}({y}) = \pm 1`):

.. math::

   c(x) = \frac{ \text{erf} \left( \frac{\mu - \log x}{\sqrt{2 \sigma^2}} \right) - \text{erf} \left( \frac{\mu - \log x_{min}}{\sqrt{2 \sigma^2}} \right) }{ -1 - \text{erf} \left( \frac{\mu - \log x_{min}}{\sqrt{2 \sigma^2}} \right) }

.. math::

   = - \frac{ \text{erf} \left( \frac{\mu - \log x}{\sqrt{2 \sigma^2}} \right) - \text{erf} \left( \frac{\mu - \log x_{min}}{\sqrt{2 \sigma^2}} \right) }{ \text{erfc} \left( \frac{\log x_{min} - \mu}{\sqrt{2 \sigma^2}} \right) }

And for :math:`x_{min} = 0` as well, we have:

.. math::

   c(x) = \frac{ \text{erf} \left( \frac{\mu - \log x}{\sqrt{2 \sigma^2}} \right) - 1 }{ -1 - 1}

.. math::

   = \frac{1}{2} \text{erfc} \left( \frac{\mu - \log x}{\sqrt{2 \sigma^2}} \right)


Inverse CDF
===========

In order to generate random numbers, we use inverse transform sampling,
which involves inverting the CDF. For arbitrary values of :math:`x_{min}` and :math:`x_{max}`,
this is:

.. math::

   r = \frac{ \text{erf} \left( \frac{\mu - \log c^\dagger(r)}{\sqrt{2 \sigma^2}} \right) - \text{erf} \left( \frac{\mu - \log x_{min}}{\sqrt{2 \sigma^2}} \right) }{ \text{erf} \left( \frac{\mu - \log x_{max}}{\sqrt{2 \sigma^2}} \right) - \text{erf} \left( \frac{\mu - \log x_{min}}{\sqrt{2 \sigma^2}} \right) }

.. math::

   r \left( \text{erf} \left( \frac{\mu - \log x_{max}}{\sqrt{2 \sigma^2}} \right) - \text{erf} \left( \frac{\mu - \log x_{min}}{\sqrt{2 \sigma^2}} \right) \right) = \text{erf} \left( \frac{\mu - \log c^\dagger(r)}{\sqrt{2 \sigma^2}} \right) - \text{erf} \left( \frac{\mu - \log x_{min}}{\sqrt{2 \sigma^2}} \right)

.. math::

   r \text{erf} \left( \frac{\mu - \log x_{max}}{\sqrt{2 \sigma^2}} \right) + (1 -r ) \text{erf} \left( \frac{\mu - \log x_{min}}{\sqrt{2 \sigma^2}} \right) = \text{erf} \left( \frac{\mu - \log c^\dagger(r)}{\sqrt{2 \sigma^2}} \right)

.. math::

   c^\dagger(r) = \text{exp} \left[ \mu - \sqrt{2 \sigma^2} \text{erf}^\dagger \left( r \text{erf} \left( \frac{\mu - \log x_{max}}{\sqrt{2 \sigma^2}} \right) + (1 - r) \text{erf} \left( \frac{\mu - \log x_{min}}{\sqrt{2 \sigma^2}} \right) \right) \right]

where :math:`\text{erf}^\dagger` is the inverse error function.

For a semi-infinite domain (:math:`x_{max} \rightarrow \infty`), this simplifies
to:

.. math::

   c^\dagger(r) = \text{exp} \left[ \mu - \sqrt{2 \sigma^2} \text{erf}^\dagger \left( (1 - r) \text{erf} \left( \frac{\mu - \log x_{min}}{\sqrt{2 \sigma^2}} \right) - r \right) \right]

And for :math:`x_{min} = 0` as well, we have:

.. math::

    c^\dagger(r) = \text{exp} \left[ \mu - \sqrt{2 \sigma^2} \text{erf}^\dagger \left( 1 - 2 r \right) \right]

In order to generate random numbers that follow a lognormal distribution,
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
