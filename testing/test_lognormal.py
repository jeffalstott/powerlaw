r"""
This test file contains tests that involve generating synthetic data and
then fitting to make sure we get the expected result for lognormal
distributions.

We can easily generate lognormal distributed samples by performing a 
multiplicative random walk:

    X_N = \prod_i  x_i

The sigma parameter for the lognormal distribution will be :math:`\sqrt{N}`.

TODO: I don't know what the mu parameter will be, so this currently
doesn't compare it to anything, just the width parameter. From what I can
tell, the mu value is usually be close to :math:`4 \sqrt{N}`, potentially
with the 4 coming from twice the random generation domain.

Also relevant to the above point, lognormals aren't super sensitive to the
mu value over ranges we might consider "typical", so fitting isn't always
great. If you want to convince yourself of this, try the following:

.. code-block::

    ln = powerlaw.Lognormal(xmin=1, xmax=1e10, parameters=[10, 5], discrete=False)
    ln2 = powerlaw.Lognormal(xmin=1, xmax=1e10, parameters=[5, 5], discrete=False)

    ln.plot_pdf()
    ln2.plot_pdf()

    plt.show() 

Remember, mu represents an *order of magnitude* of the data, and yet
doubling it barely changes the shape of the curve in the given domain.
On the other hand, try changing the sigma/width value and you will get
drastically different results.

All of this just to say, that comparing mu values from lognormal fits as
a test case is maybe difficult even if we have an expression for the theoretical
mu value for a multiplicative random walk.
"""

import unittest
import powerlaw

import numpy as np
from numpy.testing import assert_allclose


def multi_random_walk(N, mu=0):
    r"""
    A very simple multiplicative random walk.

    The range of random uniform numbers is important, but I don't know
    how to derive the exact dependence of parameters on it. For sure
    we want to include numbers above 1 and below 1, but otherwise this
    choice is arbitrary right now. Note though that if you change this,
    for example to [0.5, 1.5], then the :math:`\sigma \sim \sqrt{N}` relation
    will no longer hold.

    longdouble type is used to avoid underflow errors, since these values
    can get very small (~1e-100) for even modest N values (> 50).

    The maximum value returned from this walk will generally be of order
    one, with the vast majority of values being very small ~1e-15, so it
    is a good idea to divide by the smallest value after taking many samples
    so you can use xmin=1.

    For more information on this, see, eg.
    Bazant – 18.366 Random Walks & Diffusion – 2006 – Lecture 8: From Random Walks to Diffusion
    https://ocw.mit.edu/courses/18-366-random-walks-and-diffusion-fall-2006/982e2cfdc9a0aed0ad922773bee1eb5e_lec08_06.pdf
    """
    return np.prod(np.random.uniform(1e-10 + mu, 2 + mu, size=N).astype(np.longdouble))


def lognormal_generate(N, L):
    """
    Parameters
    ----------
    N : int
        The number of random numbers to multiply in generating each sample.

    L : int
        The number of samples to take.

    Returns
    -------
    numpy.ndarray[L]
        The data sampled from a lognormal distribution.
    """
    sample_data = np.array([multi_random_walk(N, 0) for _ in range(L)])
    sample_data /= np.min(sample_data)

    return sample_data


class TestExternalLognormal(unittest.TestCase):

    #######################################################################
    #                   CONTINUOUS LOGNORMAL
    #######################################################################

    def test_continuous(self):
        atol = 0.01
        rtol = 0.15

        # Our fit sigma should be sqrt(N)
        NArr = [25, 36, 49]
        # Number of samples to take
        L = 10000

        num_repeats = 1

        for N in NArr:
            for i in range(num_repeats):
                data = lognormal_generate(N, L)
                fit = powerlaw.Fit(data, xmin=1)

                assert_allclose([fit.lognormal.width], [np.sqrt(N)], rtol=rtol, atol=atol)


if __name__ == '__main__':
    unittest.main()
