.. powerlaw documentation master file, created by
   sphinx-quickstart on Wed Oct  8 12:12:18 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``powerlaw``
============

This is the documentation for the Python package ``powerlaw``, which provides
functions for fitting and comparing heavy-tailed distributions, primarily
based on the methods described in
`Clauset et al. (2007) <http://arxiv.org/abs/0706.1062>`_ and
`Klaus et al. (2011) <http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0019779>`_.

Academics, please cite as:

Jeff Alstott, Ed Bullmore, Dietmar Plenz. (2014). powerlaw: a Python
package for analysis of heavy-tailed distributions. PLoS ONE 9(1): e85777

Quick links
-----------
- Original paper for the library: http://arxiv.org/abs/1305.0215
- Source code: https://github.com/jeffalstott/powerlaw


Installation
------------

The package can be installed from PyPi using pip:

.. code-block:: console

    $ pip install powerlaw

Alternatively, you can install directly from the source:

.. code-block:: console

    $ git clone https://github.com/jeffalstott/powerlaw
    $ cd powerlaw
    $ pip install .

This library depends on the usual scientific computing libraries that you
probably already have installed: ``numpy``, ``scipy``, ``matplotlib``, and
``mpmath``.

The package ``tqdm`` is used for creating progress bars.

The requirement of ``mpmath`` will be dropped if/when the scipy functions
``gamma``, ``gammainc`` and ``gammaincc`` are updated to have sufficient numerical
accuracy for negative numbers.

See the `powerlaw home page <https://github.com/jeffalstott/powerlaw>`_ for more
information and examples.

Basic usage
-----------

The most basic use of this library is to fit some data, extract parameters,
and make comparisons to other distributions:

.. code-block::

    import powerlaw
    import numpy as np

    data = np.array([1.7, 3.2 ...]) # data can be list or numpy array
    fit = powerlaw.Fit(data)

    print(fit.power_law.alpha)
    print(fit.power_law.xmin)

    R, p = fit.distribution_compare('power_law', 'lognormal')

You can also plot various results easily using ``matplotlib``:

.. code-block::

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    fit.plot_pdf(ax=ax, label='PDF')
    fit.power_law.plot_pdf(ax=ax, label='Power law fit')

    plt.legend()
    plt.show()

For more in-depth usage, see the various `tutorials <tutorials_top.html>`_,
`API pages <api.html>`_, or the `original manuscript <http://arxiv.org/abs/1305.0215>`_
(though the documentation will be more up-to-date).


Indices and tables
------------------

.. toctree::
    :maxdepth: 1

    tutorials_top
    api
    contributing

* :ref:`genindex`
* :ref:`search`

