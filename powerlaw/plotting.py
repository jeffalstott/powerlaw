"""
Methods for plotting distributions.
"""
import matplotlib.pyplot as plt

import numpy as np
from numpy import nan

from .statistics import *

def plot_ccdf(data, xmin=None, xmax=None, ax=None, **kwargs):
    """
    Plots the complementary cumulative distribution function (CDF) of the data
    to a new figure or to axis ax if provided.

    Parameters
    ----------
    data : array_like
        The data to compute the CCDF for.

    xmin : float, optional
        Minimum value of the CCDF. If None, uses the smallest value in the data.

    xmax : float, optional
        Maximum value of the CCDF. If None, uses the largest value in the data.

    ax : matplotlib axis, optional
        The axis to which to plot. If None, a new figure is created.

    kwargs
        Other keyword arguments are passed to `matplotlib.pyplot.plot()`.

    Returns
    -------
    ax : matplotlib axis
        The axis to which the plot was made.
    """
    # Compute the cdf
    bins, CCDF = ccdf(data, xmin=xmin, xmax=xmax)

    # If we don't have an axis, create one
    if not ax:
        plt.plot(bins, CCDF, **kwargs)
        ax = plt.gca()

    else:
        ax.plot(bins, CCDF, **kwargs)

    ax.set_xscale("log")
    ax.set_yscale("log")

    return ax


def plot_cdf(data, xmin=None, xmax=None, ax=None, **kwargs):
    """
    Plots the cumulative distribution function (CDF) of the data to a new
    figure or to axis ax if provided.

    Parameters
    ----------
    data : array_like
        The data to compute the CDF for.

    xmin : float, optional
        Minimum value of the CDF. If None, uses the smallest value in the data.

    xmax : float, optional
        Maximum value of the CDF. If None, uses the largest value in the data.

    ax : matplotlib axis, optional
        The axis to which to plot. If None, a new figure is created.

    kwargs
        Other keyword arguments are passed to `matplotlib.pyplot.plot()`.

    Returns
    -------
    ax : matplotlib axis
        The axis to which the plot was made.
    """
    # Compute the cdf
    bins, CDF = cdf(data, xmin=xmin, xmax=xmax)

    # If we don't have an axis, create one
    if not ax:
        plt.plot(bins, CDF, **kwargs)
        ax = plt.gca()

    else:
        ax.plot(bins, CDF, **kwargs)

    ax.set_xscale("log")
    ax.set_yscale("log")

    return ax


def plot_pdf(data, xmin=None, xmax=None, linear_bins=False, bins=None, ax=None, **kwargs):
    """
    Plots the probability density function (PDF) to a new figure or to axis ax
    if provided.

    Parameters
    ----------
    data : list or array
        The data to compute the PDF for.

    xmin : float, optional
        Minimum value of the PDF. If None, uses the smallest value in the data.

    xmax : float, optional
        Maximum value of the PDF. If None, uses the largest value in the data.

    linear_bins : bool, optional
        Whether to use linearly spaced bins, as opposed to logarithmically
        spaced bins (recommended for log-log plots).

    bins : array_like, optional
        The bins within which to compute the PDF.

        If not provided, will be generated based on the range of the data.
        By default, the bins will be logarithmically spaced, but can be
        linear if `linear_bins=True`.

    ax : matplotlib axis, optional
        The axis to which to plot. If None, a new figure is created.

    kwargs
        Other keyword arguments are passed to `matplotlib.pyplot.plot()`.

    Returns
    -------
    ax : matplotlib axis
        The axis to which the plot was made.
    """
    # Compute the pdf
    edges, hist = pdf(data, xmin=xmin, xmax=xmax, linear_bins=linear_bins, bins=bins)

    bin_centers = (edges[1:] + edges[:-1]) / 2
    # Make it nan instead of 0 so we don't have lots of vertical lines
    # on the log plot.
    hist[hist == 0] = nan

    if not ax:
        plt.plot(bin_centers, hist, **kwargs)
        ax = plt.gca()

    else:
        ax.plot(bin_centers, hist, **kwargs)

    # Only set log scale on x if we have logarithmic bins
    if not linear_bins:
        ax.set_xscale("log")

    ax.set_yscale("log")

    return ax
