import matplotlib.pyplot as plt

import numpy as np
from numpy import nan

from .statistics import *

def plot_ccdf(data, ax=None, survival=False, **kwargs):
    """
    Plots the complementary cumulative distribution function (CDF) of the data
    to a new figure or to axis ax if provided.

    Parameters
    ----------
    data : list or array
    ax : matplotlib axis, optional
        The axis to which to plot. If None, a new figure is created.
    survival : bool, optional
        Whether to plot a CDF (False) or CCDF (True). True by default.

    Returns
    -------
    ax : matplotlib axis
        The axis to which the plot was made.
    """
    return plot_cdf(data, ax=ax, survival=True, **kwargs)

def plot_cdf(data, ax=None, survival=False, **kwargs):
    """
    Plots the cumulative distribution function (CDF) of the data to a new
    figure or to axis ax if provided.

    Parameters
    ----------
    data : list or array
    ax : matplotlib axis, optional
        The axis to which to plot. If None, a new figure is created.
    survival : bool, optional
        Whether to plot a CDF (False) or CCDF (True). False by default.

    Returns
    -------
    ax : matplotlib axis
        The axis to which the plot was made.
    """
    bins, CDF = cdf(data, survival=survival, **kwargs)
    if not ax:
        plt.plot(bins, CDF, **kwargs)
        ax = plt.gca()
    else:
        ax.plot(bins, CDF, **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    return ax


def plot_pdf(data, ax=None, linear_bins=False, **kwargs):
    """
    Plots the probability density function (PDF) to a new figure or to axis ax
    if provided.

    Parameters
    ----------
    data : list or array
    ax : matplotlib axis, optional
        The axis to which to plot. If None, a new figure is created.
    linear_bins : bool, optional
        Whether to use linearly spaced bins (True) or logarithmically
        spaced bins (False). False by default.

    Returns
    -------
    ax : matplotlib axis
        The axis to which the plot was made.
    """
    if 'bins' in kwargs.keys():
        bins = kwargs.pop('bins')
    else:
        bins = None

    edges, hist = pdf(data, linear_bins=linear_bins, bins=bins, **kwargs)
    bin_centers = (edges[1:]+edges[:-1])/2.0
    hist[hist==0] = nan
    if not ax:
        plt.plot(bin_centers, hist, **kwargs)
        ax = plt.gca()
    else:
        ax.plot(bin_centers, hist, **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    return ax
