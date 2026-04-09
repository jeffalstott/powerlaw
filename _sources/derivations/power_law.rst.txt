Power Law Derivations
---------------------

This page covers the analytics and derivations relevant to the power
law distribution.

Probability distribution function (PDF)
=======================================

The PDF for a power law distribution has the form:

.. math::

    p(x) = C x^{-\alpha}

Defined over the domain :math:`[x_{min}, x_{max}]`, and for :math:`\alpha \ne 1`,
this has normalization constant:

.. math::

   C = \left[ \int_{x_{min}}^{x_{max}} x^{-\alpha} dx \right]^{-1}
   = (\alpha - 1) \left[ x_{max}^{1 - \alpha} - x_{min}^{1-\alpha} \right]^{-1}

Typically we deal with :math:`\alpha` in the range :math:`(1, 3]`, though
for any :math:`\alpha > 1` this normalization constant is finite even if we
take a semi-infinite distribution, :math:`x \in [x_{min}, \infty]`. In this
case, we have normalization:

.. math::

    C = (\alpha - 1) x_{min}^{\alpha - 1}

When :math:`\alpha < 1`, this normalization constant is finite so long as
there is a well-defined value for :math:`x_{max}`.

The case of :math:`\alpha = 1` is quite rare, and leads to a logarithmic
normalization:

.. math::

    C = \log{\frac{x_{max}}{x_{min}}}

Given that this only holds when alpha is *exactly* one, this case is
not treated in this package, though there are some considerations that
must be taken when :math:`\alpha` is close to one to deal with this
discontinuity.


Cumulative distribution function (CDF)
=======================================

The CDF for a power law distribution is:

.. math::
    
    c(x) = C \int_{x_{min}}^{x} {x'}^{-\alpha} dx'

.. math::

    = \frac{C}{\alpha - 1} \left(x^{1 - \alpha} - x_{min}^{1 - \alpha} \right)

.. math::

    = \frac{x^{1 - \alpha} - x_{min}^{1 - \alpha} }{ x_{max}^{1 - \alpha} - x_{min}^{1-\alpha}}

For a semi-infinite domain (:math:`x_{max} \rightarrow \infty`) and :math:`\alpha > 1`, this simplifies
to:

.. math::

    c(x) = 1 - \left( \frac{x}{x_{min}} \right)^{1 - \alpha}

As mentioned above, for :math:`\alpha < 1`, we necessarily have some finite
value for :math:`x_{max}`.

Inverse CDF
===========

In order to generate random numbers, we use inverse transform sampling,
which involves inverting the CDF. For an arbitrary value of :math:`x_{max}`,
this is:

.. math::

    r = \frac{ \left( c^{\dagger}(r) \right)^{1 - \alpha} - x_{min}^{1 - \alpha} }{ x_{max}^{1 - \alpha} - x_{min}^{1-\alpha}}

.. math::

    r \left( x_{max}^{1 - \alpha} - x_{min}^{1-\alpha} \right) + x_{min}^{1 - \alpha} = \left( c^{\dagger}(r) \right)^{1 - \alpha}

.. math::

    r x_{max}^{1 - \alpha} + (1 - r) x_{min}^{1-\alpha} = \left( c^{\dagger}(r) \right)^{1 - \alpha}

.. math::

    c^{\dagger}(r) = \left[ r x_{max}^{1 - \alpha} + (1 - r) x_{min}^{1-\alpha} \right]^{\frac{1}{1 - \alpha}}

We use the above expression directly when :math:`\alpha < 1`.

For a semi-infinite domain (:math:`x_{max} \rightarrow \infty`) and :math:`\alpha > 1`, this simplifies
to:

.. math::

    c^{\dagger}(r) = x_{min} (1 - r)^{\frac{1}{1 - \alpha}}

In order to generate random numbers that follow a power law distribution,
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

The above choice is only available for power laws with :math:`\alpha > 1`; power laws
with :math:`\alpha < 1` must have a finite :math:`x_{max}` value, and therefore
we have to use the full expression (though we can just use uniform random numbers
in the range :math:`[0, 1)`).
