"""
This test file contains tests that involve generating random numbers
from distributions.
"""

import unittest
import pytest

import powerlaw

import numpy as np

class TestRandomGeneration(unittest.TestCase):

    def test_rng_size(self):
        """
        Make sure that we actually generate the number of points that are
        requested.
        """
        alpha = 1.5
        N = 2000

        pl = powerlaw.Power_Law(xmin=1, xmax=1e6, parameters=[alpha])
        data = pl.generate_random(N)

        assert len(data) == N


    def test_rng_bounds_continuous(self):
        """
        Make sure that the randomly generated numbers are within the
        specified xmin and xmax of the distribution for continuous numbers.

        There is a separate function for power laws since they have a
        discontinuity in the exponent at 1.
        """
        distribution_arr = [
                            powerlaw.Exponential,
                            powerlaw.Stretched_Exponential,
                            powerlaw.Truncated_Power_Law,
                            powerlaw.Lognormal,
                            powerlaw.Lognormal_Positive
                           ]

        # These are just reasonable parameter choices for each distribution,
        # but the specific value shouldn't matter
        parameter_arr = [
                         [0.01], # [Lambda]
                         [0.01, 0.5], # [Lambda, beta]
                         [1.5, 0.01], # [alpha, Lambda]
                         [1, 1.5], # [mu, sigma]
                         [1, 1.5], # [mu, sigma]
                        ]
        N = 2000
        xmin = 1
        xmax = 1e3

        for i in range(len(distribution_arr)):
            print(str(distribution_arr[i]))

            pl = distribution_arr[i](xmin=xmin, xmax=xmax, parameters=parameter_arr[i])
            data = pl.generate_random(N)

            assert all(data <= xmax), f'{str(distribution_arr[i])}, upper bound'
            assert all(data >= xmin), f'{str(distribution_arr[i])}, lower bound'


    def test_rng_bounds_discrete(self):
        """
        Make sure that the randomly generated numbers are within the
        specified xmin and xmax of the distribution for discrete numbers.

        There is a separate function for power laws since they have a
        discontinuity in the exponent at 1.
        """
        distribution_arr = [
                            powerlaw.Exponential,
                            powerlaw.Stretched_Exponential,
                            powerlaw.Truncated_Power_Law,
                            powerlaw.Lognormal,
                            powerlaw.Lognormal_Positive
                           ]

        # These are just reasonable parameter choices for each distribution,
        # but the specific value shouldn't matter
        parameter_arr = [
                         [0.01], # [Lambda]
                         [0.01, 0.5], # [Lambda, beta]
                         [1.5, 0.01], # [alpha, Lambda]
                         [1, 1.5], # [mu, sigma]
                         [1, 1.5], # [mu, sigma]
                        ]
        N = 2000
        xmin = 1
        xmax = 1e3

        for i in range(len(distribution_arr)):
            print(str(distribution_arr[i]))

            pl = distribution_arr[i](xmin=xmin, xmax=xmax, parameters=parameter_arr[i], discrete=True)
            data = pl.generate_random(N)

            assert all(data <= xmax), f'{str(distribution_arr[i])}, upper bound'
            assert all(data >= xmin), f'{str(distribution_arr[i])}, lower bound'


    def test_rng_upper_bound_power_law_alpha_0p5_to_0p9(self):
        """
        Make sure that the randomly generated numbers are within the
        specified xmin and xmax of the distribution.

        This is split into exponents above 1 and below 1, since there is
        the discontinuity there.

        They are also split into upper and lower bounds, primarily to debug
        an issue with an old version (v1.5.0) of powerlaw. Could eventually
        be combined if desired.
        """
        alpha_arr = np.linspace(0.5, 0.9, 3)
        N = 2000
        xmin = 1
        xmax = 1e3

        for alpha in alpha_arr:

            pl = powerlaw.Power_Law(xmin=xmin, xmax=xmax, parameters=[alpha])
            data = pl.generate_random(N)

            assert all(data <= xmax)


    def test_rng_upper_bound_power_law_alpha_1p1_to_2p5(self):
        """
        Make sure that the randomly generated numbers are within the
        specified xmin and xmax of the distribution.

        This is split into exponents above 1 and below 1, since there is
        the discontinuity there.
        """
        alpha_arr = np.linspace(1.1, 2.5, 5)
        N = 2000
        xmin = 1
        xmax = 1e6

        for alpha in alpha_arr:

            pl = powerlaw.Power_Law(xmin=xmin, xmax=xmax, parameters=[alpha])
            data = pl.generate_random(N)

            assert all(data <= xmax)


    def test_rng_lower_bound_power_law_alpha_0p5_to_0p9(self):
        """
        Make sure that the randomly generated numbers are within the
        specified xmin and xmax of the distribution.

        This is split into exponents above 1 and below 1, since there is
        the discontinuity there.

        They are also split into upper and lower bounds.
        """
        alpha_arr = np.linspace(0.5, 0.9, 3)
        N = 2000
        xmin = 1
        xmax = 1e3

        for alpha in alpha_arr:

            pl = powerlaw.Power_Law(xmin=xmin, xmax=xmax, parameters=[alpha])
            data = pl.generate_random(N)

            assert all(data >= xmin)

    def test_rng_lower_bound_power_law_alpha_1p1_to_2p5(self):
        """
        Make sure that the randomly generated numbers are within the
        specified xmin and xmax of the distribution.

        This is split into exponents above 1 and below 1, since there is
        the discontinuity there.
        """
        alpha_arr = np.linspace(1.1, 2.5, 5)
        N = 2000
        xmin = 1
        xmax = 1e3

        for alpha in alpha_arr:

            pl = powerlaw.Power_Law(xmin=xmin, xmax=xmax, parameters=[alpha])
            data = pl.generate_random(N)

            assert all(data >= xmin)


if __name__ == '__main__':
    unittest.main()
