Exponential Derivations
---------------------

This page covers the analytics and derivations relevant to the exponential
distribution.

Probability distribution function (PDF)
=======================================

The PDF for an exponential distribution has the form:

.. math::

    p(x) = C e^{-\lambda x}

Defined over the domain :math:`[x_{min}, x_{max}]`, this has normalization constant:

.. math::

   C = \left[ \int_{x_{min}}^{x_{max}} e^{-\lambda x} dx \right]^{-1}
   =  - \lambda \left[ e^{-\lambda x_{max}} - e^{-\lambda x_{min}} \right]^{-1}

For a semi-infinite domain (:math:`x_{max} \rightarrow \infty`), this simplifies
to:

.. math::

    C = \lambda e^{\lambda x_{min}}

Note that this can cause overflow and underflow issues if we store this
normalization constant as a separate value from the unnormalized PDF values.

Cumulative distribution function (CDF)
=======================================

The CDF for an exponential distribution is:

.. math::
    
    c(x) = C \int_{x_{min}}^{x} e^{-\lambda x'} dx'

.. math::

    = - \frac{C}{\lambda} \left[ e^{-\lambda x} - e^{-\lambda x_{min}} \right]

.. math::

    =  \frac{ e^{-\lambda x} - e^{-\lambda x_{min}} }{ e^{-\lambda x_{max}} - e^{-\lambda x_{min}} }

For a semi-infinite domain (:math:`x_{max} \rightarrow \infty`), this simplifies
to:

.. math::

    c(x) = 1 - e^{-\lambda (x - x_{min})}

Inverse CDF
===========

In order to generate random numbers, we use inverse transform sampling,
which involves inverting the CDF. For an arbitrary value of :math:`x_{max}`,
this is:

.. math::

    r =  \frac{ e^{-\lambda c^\dagger(r)} - e^{-\lambda x_{min}} }{ e^{-\lambda x_{max}} - e^{-\lambda x_{min}} }

.. math::

    r \left( e^{-\lambda x_{max}} - e^{-\lambda x_{min}} \right) =  e^{-\lambda c^\dagger(r)} - e^{-\lambda x_{min}}

.. math::

    r e^{-\lambda x_{max}} + (1 - r) e^{-\lambda x_{min}} = e^{-\lambda c^{\dagger}(r)}

.. math::

    c^\dagger(r) = - \frac{1}{\lambda} \log \left( r e^{-\lambda x_{max}} + (1 - r) e^{-\lambda x_{min}} \right)

For a semi-infinite domain (:math:`x_{max} \rightarrow \infty`), this simplifies
to:

.. math::

    c^\dagger(r) = - \frac{1}{\lambda} \log \left( (1 - r) e^{-\lambda x_{min}} \right)

.. math::

    c^\dagger(r) = x_{min} - \frac{1}{\lambda} \log \left( 1 - r \right)

In order to generate random numbers that follow an exponential distribution,
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
