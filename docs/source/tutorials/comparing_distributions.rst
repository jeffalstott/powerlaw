Comparing distributions
=======================

As introduced in `the comparison between this library and others <other_libraries.html>`_
one of the main advantages of having properly defined probability
distributions is being able to easily make comparisons between different
candidate distributions.

The goodness of fit of a distributions must be evaluated before concluding
that a power law (or another distribution) is a good description of the data.
The goodness of fit for each distribution can be considered individually ---
using bootstrapping and the Kolmogorov-Smirnov test to generate a p-value ---
or by comparison to other distributions --- using loglikelihood ratios to
identify which of two fits is better [1]. There are several reasons, both
practical and philosophical, to focus on the latter, comparative tests. 

Practically, bootstrapping is more computationally intensive and loglikelihood
ratio tests are faster. Philosophically, it is frequently insufficient and
unnecessary to answer the question of whether a distribution "really" follows
a power law. Instead the question is whether a power law is the best
description available. In such a case, the knowledge that a bootstrapping
test has passed is insufficient; bootstrapping could indeed find that a power
law distribution would produce a given dataset with sufficient likelihood,
but a comparative test could identify that a lognormal fit could have
produced it with even greater likelihood. On the other hand, the knowledge
that a bootstrapping test has failed may be unnecessary; real world systems
have noise, and so few empirical phenomena could be expected to follow a
power law with the perfection of a theoretical distribution. Given enough
data, an empirical dataset with any noise or imperfections will always fail
a bootstrapping test for any theoretical distribution. If one keeps absolute
adherence to the exact theoretical distribution, one can enter the tricky
position of passing a bootstrapping test, but only with few enough data [2] 

Thus, it is generally more sound and useful to compare the fits of many
candidate distributions, and identify which one fits the best. This is 
easily done using ``powerlaw``:

.. code-block::

    # Let's use the blackouts dataset 
    data = powerlaw.load_test_dataset('blackouts')

    fit = powerlaw.Fit(data)

    # Plot the data and two candidate distributions
    fit.plot_pdf(label='Data')
    fit.power_law.plot_pdf(label='Power law fit')
    fit.exponential.plot_pdf(label='Exponential fit')

    ...

    # See which distribution fits the data better
    R, p = fit.distribution_compare('power_law', 'exponential',
                                    normalized_ratio=True)

    print(R, p)

.. figure:: ../images/blackouts_compare.png

    Comparison of the PDFs for power law and exponential fits for ``blackouts`` dataset.

This function :meth:`Fit.distribution_compare` gives us two values back:
the loglikelihood ratio of the two distributions, and the significance 
value of that ratio. For this case, those values are:

.. code-block:: output

    (1.4314804849576281, 0.15229255604426545)

Since the ratio is positive, that means that the result of the test may support
the first distribution (power law) over the second distribution (exponential).
If the value were negative, we would have the opposite possibility.
To tell if the test actually supports one distribution over
the other, we look at the magnitude of the ratio, or even better, at the
significance level. In this particular case, we have a pretty large
p value, and using most standard choices of significance level -- usually
:math:`0.01` or :math:`0.05` --- we would fail to reject the null hypothesis.
That is, the power law distribution is not vastly better than the exponential,
and we don't have strong enough evidence to conclude that this is really
a power law. If this were our real data that we want to analyze, we should
think about gathering more data in order to say more confidently whether this is
a power law or not.

The exponential is often the first alternative distribution one should check
when looking at heavy tailed data. The reason is definitional: the typical
quantitative definition of a "heavy-tail" is that it is `not` exponentially
bounded [3]. Thus if a power law is not a better fit than an exponential
distribution (as in the above example) there is scarce ground for considering
the distribution to be heavy-tailed at all, let alone a power law.

Let's look at another example: the ``words`` dataset.

.. code-block::

    data = powerlaw.load_test_dataset('word')

    # Remember this is a discrete dataset
    fit = powerlaw.Fit(data, discrete=True)

    ...

    # See which distribution fits the data better
    R, p = fit.distribution_compare('power_law', 'exponential',
                                    normalized_ratio=True)

    print(R, p)

.. figure:: ../images/words_compare.png

    Comparison of the PDFs for power law and exponential fits for ``words`` dataset.

It's pretty clear from the plot that the power law is a much better fit
than an exponential for this distribution; indeed the log likelihood
comparison agrees:

.. code-block:: output

    (9.135914718777004, 6.485614241379349e-20)

Since our p value is (much) smaller than any reasonable significance value,
we have pretty strong evidence that this distribution follows a power law
much more than an exponential. 

However, the exponential distribution is only the minimum alternative
candidate distribution to consider when describing a probability distribution.
Among the other common heavy-tailed distributions are the exponentially
truncated power law, the lognormal, and the stretched exponential (Weibull)
distributions.

Of course, one can define their own custom distribution beyond standard
choices, though care must be taken to avoid the same problem faced by
bootstrapping: there will always be another distribution that fits the data
better, until one arrives at a distribution that describes only the exact
values and frequencies observed in the dataset (overfitting). Indeed, this
process of overfitting can begin even with very simple distributions; while
the power law has only one parameter to serve as a degree of freedom for fitting,
the truncated power law and the alternative heavy-tailed distributions have
two parameters, and thus a fitting advantage. The overfitting scenario can
be avoided by incorporating generative mechanisms into the candidate
distribution selection process.

The observed data always come from some source, and there must be some
generative mechanisms behind this source. If there is a plausible domain-specific
mechanism for creating the data that would yield a particular candidate
distribution, then that candidate distribution should be considered first-and-
foremost for fitting. If there is no such hypothesis for how a candidate distribution could be created there is much less reason to use it to describe the dataset. 

Perhaps the simplest generative mechanism is the accumulation of independent
random variables: the central limit theorem. When random variables are summed,
the result is the normal distribution. However, when positive random variables
are multiplied, the result is the lognormal distribution, which is quite
heavy-tailed. If the generative mechanism for the lognormal is plausible
for the domain, the lognormal is frequently just as good a fit as the power
law, if not better. Let's compare the two for the ``words`` dataset.

.. code-block::

    # See which distribution fits the data better
    R, p = fit.distribution_compare('power_law', 'lognormal',
                                    normalized_ratio=True)

    print(R, p)

.. code-block:: output
    
    (0.0636808247240506, 0.9492243734054262)

Based on our distribution comparison, indeed there is no evidence to suggest
that a lognormal distribution is better or worse than a power law for this
data.

.. figure:: ../images/words_compare_lognormal.png

    Comparison of the CCDFs for power law and lognormal fits for ``words`` dataset.

There are domains in which the power law distribution is a superior fit to
the lognormal [eg. 2]. However, difficulties in distinguishing the power law
from the lognormal are common and well-described, and similar issues apply
to the stretched exponential and other heavy-tailed distributions [4-6]. If
faced with such difficulties, it is a good idea to reevaluate the
justification behind fitting each distribution for your specific data.
If you still cannot distinguish between feasible candidate distributions using
the loglikelihood ratio test, you might need to collect more data or try
other experiments.


Nested distributions
--------------------

Comparing the likelihoods of distributions that are nested versions of
each other requires a particular calculation for the resulting p-value
[1]. This calculation mode can be enabled explicitly with the ``nested``
keyword in :meth:`powerlaw.Fit.distribution_compare` or
:meth:`powerlaw.loglikelihood_ratio`.

.. code-block::

    # Specify whether the distributions are nested
    fit.distribution_compare(’power_law’, ’truncated_power_law’, nested=True)
    
    # Default behavior; let powerlaw decide if they are nested based on name
    fit.distribution_compare(’power_law’, ’truncated_power_law’)

.. code-block::

    # Manually compare two distributions without the fit object; can still
    # specify the nested keyword.
    data = powerlaw.load_test_dataset('fires')

    loglikelihoods_pl = powerlaw.Power_Law(data=data).loglikelihoods()
    loglikelihoods_exp = powerlaw.Exponential(data=data).loglikelihoods()

    powerlaw.loglikelihood_ratio(loglikelihoods_pl, loglikelihoods_exp, nested=False)

When this keyword is explicitly set, the nested calculation method will only
be used if the name of one distribution is contained within another, for
example, if you are comparing ``'power_law'`` with ``'truncated_power_law'``.

With this in mind, if you create new distributions to add to the library,
make sure to name them appropriately if they are a specific case of another
distribution. 


References
----------

[1] Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law
distributions in empirical data. SIAM Review, 51(4), 661–703.
https://doi.org/10.1137/070710111

[2] Klaus, A., Yu, S., & Plenz, D. (2011). Statistical Analyses Support
Power Law Distributions Found in Neuronal Avalanches. PLOS ONE, 6(5),
e19779. https://doi.org/10.1371/journal.pone.0019779

[3] Asmussen Sr (2003) Applied probability and queues. Berlin: Springer

[4] Malevergne, Y., Pisarenko, V., & Sornette, D. (2011). Testing the
Pareto against the lognormal distributions with the uniformly most powerful
unbiased test applied to the distribution of cities. Physical Review E,
83(3), 036111. https://doi.org/10.1103/PhysRevE.83.036111

[5] Malevergne, Y., Pisarenko, V. F., & Sornette, D. (2003). Empirical
Distributions of Log-Returns: Between the Stretched Exponential and the
Power Law? arXiv.
https://doi.org/10.48550/arXiv.physics/0305089

[6] Mitzenmacher, M. (2004). A Brief History of Generative Models for Power
Law and Lognormal Distributions. Internet Mathematics, 1(2), 226–251.
https://doi.org/10.1080/15427951.2004.10129088

