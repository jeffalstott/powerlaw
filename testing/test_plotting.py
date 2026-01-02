"""
This test file contains tests related to plotting functions.
"""
import unittest
import powerlaw
from numpy.testing import assert_allclose
import numpy as np


class TestPlotPDF(unittest.TestCase):
    def test_custom_bins(self):

        import numpy as np
        import powerlaw

        import matplotlib.pyplot as plt

        data = 1. / np.random.power(4., 1000)
        fit = powerlaw.Fit(data)

        # ax1 = fit.plot_pdf()
        plt.figure()
        bins = 2
        ax = fit.plot_pdf(marker="*", bins=bins)
        line = ax.lines[0]
        assert len(line.get_xdata()) == bins
        plt.close()

        plt.figure()
        bins = 10
        ax = fit.plot_pdf(marker="*", bins=bins)
        line = ax.lines[0]
        assert len(line.get_xdata()) == bins
        plt.close()
