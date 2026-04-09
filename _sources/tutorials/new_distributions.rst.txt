Adding New Distributions
========================

The tools of this library become more and more useful as more candidate
distributions are implemented, and as such, we encourage users to implement
distributions that they would like to use. Lots of effort has been put into
the package structure to make this as easy as possible.

Of course any new distribution should extend the :class:`powerlaw.Distribution`
class, and the first three variables you should assign are ``name``, ``parameter_names``,
and ``DEFAULT_PARAMETER_RANGES``.

.. code-block::

    class NewDistribution(Distribution):

        # This is the name of this type of distribution, that you would
        # use in eg. distribution_compare
        name = 'dist_name'

        # This should be a list with the names of parameters.
        parameter_names = ['parameter1', 'parameter2']

        # This should be a dictionary where each key is a parameter name
        # (defined above) and each value is a range [lower_bound, upper_bound].
        DEFAULT_PARAMETER_RANGES = {"parameter1": [x1, x2],
                                    "parameter2": [y1, y2]}


In most cases, you don't need to define a constructor (``__init__``) function
unless you want to your class to have some distribution-specific keywords
(see :class:`powerlaw.Power_Law` for an example) or do some non-standard
initialization.

You should next define the core of any distribution class: the PDF, CDF, and
PDF normalizer(s). These functions should make use of the parameters 
defined for the distribution, by directly accessing them from ``self``.

.. code-block::

    class NewDistribution(Distribution):

        ...

        def _pdf_base_function(self, x):
            """
            The (unnormalized) PDF.
            """
            # For example, for an exponential:
            # return np.exp(-self.Lambda * x)

        def _cdf_base_function(self, x):
            """
            The (unnormalized) CDF.
            """
            # For example, for an exponential:
            # return 1 - np.exp(-self.Lambda * x)

        @property
        def _pdf_continuous_normalizer(self):
            """
            Normalization constant :math:`A` for the continuous PDF such
            that :math:`p(x) = A q(x)` where :math:`p` is the proper
            PDF and :math:`q` is the base function.
            """
            # For example, for an exponential:
            # return self.Lambda * np.exp(self.Lambda * self.xmin)

        @property
        def _pdf_discrete_normalizer(self):
            """
            Normalization constant :math:`A` for the discrete PDF such
            that :math:`p(x) = A q(x)` where :math:`p` is the proper
            PDF and :math:`q` is the base function.

            If no exact expression exists for this, you can remove this
            function.
            """

Most of the work is done at this point! Next, you need to define a way to
initialize the parameters based on the data. The function ``generate_initial_parameters``
should take as input the data (which can be presumed to exist and already
be cropped to the defined region for the distribution) and return a 
dictionary with the parameters and their values.

.. code-block::

    class NewDistribution(Distribution):

        ...

        def generate_initial_parameters(self, data):
            """
            Generate initial values for parameters based on the provided
            data.
            """
            params = {}

            # For example, for an exponential:
            # params["Lambda"] = 1 / np.mean(data)

            return params


Lastly, you should add a function for generating random numbers from your
distribution. If you aren't sure how to derive this function, take a look
at resources on `inverse transform sampling <https://en.wikipedia.org/wiki/Inverse_transform_sampling>`_.

.. code-block::

    class NewDistribution(Distribution):

        ...

        def generate_random(self, r):
            """
            Transform a uniform random variable :math:`r` into a random
            variable sampled from this distribution.
            """
            # For example, for an exponential:
            # return self.xmin - (1 / self.Lambda) * np.log(1 - r)

By defining the functions above, you may have finished implementing your new
distribution, and can test it with the rest of the library. For more complex
implementations, you might need to overload (reimplement) some methods from
the parent ``Distribution`` class. For example, the ``Power_Law`` class
overloads the ``fit`` method since there exists an analytical expression for
the MLE to a given dataset, and therefore we might not want to run the
numerical fitting sometimes.

If you're struggling to implement a distribution, feel free to discuss with
the maintainers and other users by creating an Issue on the repository.
