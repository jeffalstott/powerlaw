"""
Methods to compute statistical quantities of distributions.
"""
import numpy as np
from numpy import nan

def nested_loglikelihood_ratio(loglikelihoods1, loglikelihoods2, **kwargs):
    """
    Calculates a loglikelihood ratio and the p-value for testing which of two
    probability distributions is more likely to have created a set of
    observations. Assumes one of the probability distributions is a nested
    version of the other.

    Parameters
    ----------
    loglikelihoods1 : list or array
        The logarithms of the likelihoods of each observation, calculated from
        a particular probability distribution.
    loglikelihoods2 : list or array
        The logarithms of the likelihoods of each observation, calculated from
        a particular probability distribution.
    nested : bool, optional
        Whether one of the two probability distributions that generated the
        likelihoods is a nested version of the other. True by default.
    normalized_ratio : bool, optional
        Whether to return the loglikelihood ratio, R, or the normalized
        ratio R/sqrt(n*variance)

    Returns
    -------
    R : float
        The loglikelihood ratio of the two sets of likelihoods. If positive,
        the first set of likelihoods is more likely (and so the probability
        distribution that produced them is a better fit to the data). If
        negative, the reverse is true.
    p : float
        The significance of the sign of R. If below a critical value
        (typically .05) the sign of R is taken to be significant. If above the
        critical value the sign of R is taken to be due to statistical
        fluctuations.
    """
    return loglikelihood_ratio(loglikelihoods1, loglikelihoods2,
            nested=True, **kwargs)


def loglikelihood_ratio(loglikelihoods1, loglikelihoods2,
        nested=False, normalized_ratio=False):
    """
    Calculates a loglikelihood ratio and the p-value for testing which of two
    probability distributions is more likely to have created a set of
    observations.

    Parameters
    ----------
    loglikelihoods1 : list or array
        The logarithms of the likelihoods of each observation, calculated from
        a particular probability distribution.
    loglikelihoods2 : list or array
        The logarithms of the likelihoods of each observation, calculated from
        a particular probability distribution.
    nested : bool, optional
        Whether one of the two probability distributions that generated the
        likelihoods is a nested version of the other. False by default.
    normalized_ratio : bool, optional
        Whether to return the loglikelihood ratio, R, or the normalized
        ratio R/sqrt(n*variance)

    Returns
    -------
    R : float
        The loglikelihood ratio of the two sets of likelihoods. If positive,
        the first set of likelihoods is more likely (and so the probability
        distribution that produced them is a better fit to the data). If
        negative, the reverse is true.
    p : float
        The significance of the sign of R. If below a critical value
        (typically .05) the sign of R is taken to be significant. If above the
        critical value the sign of R is taken to be due to statistical
        fluctuations.
    """
    from numpy import sqrt
    from scipy.special import erfc

    n = float(len(loglikelihoods1))

    if n==0:
        R = 0
        p = 1
        return R, p
    from numpy import asarray
    loglikelihoods1 = asarray(loglikelihoods1)
    loglikelihoods2 = asarray(loglikelihoods2)

    #Clean for extreme values, if any
    from numpy import inf, log
    from sys import float_info
    min_val = log(10**float_info.min_10_exp)
    loglikelihoods1[loglikelihoods1==-inf] = min_val
    loglikelihoods2[loglikelihoods2==-inf] = min_val

    R = sum(loglikelihoods1-loglikelihoods2)

    from numpy import mean
    mean_diff = mean(loglikelihoods1)-mean(loglikelihoods2)
    variance = sum(
            ( (loglikelihoods1-loglikelihoods2) - mean_diff)**2
            )/n

    if nested:
        from scipy.stats import chi2
        p = 1 - chi2.cdf(abs(2*R), 1)
    else:
        p = erfc( abs(R) / sqrt(2*n*variance))

    if normalized_ratio:
        R = R/sqrt(n*variance)

    return R, p


# Both cdf and ccdf used to have the keyword 'survival' which controlled
# whether you get the cdf or ccdf, but since we already have two functions
# having that is redundant and confusing.
def cdf(data, xmin=None, xmax=None):
    """
    Compute the cumulative distribution function (CDF) of the data.

    Parameters
    ----------
    data : list or array
        The data to compute the CDF for.

    xmin : int or float, optional
        The minimum data size to include. Values less than xmin are excluded.

    xmax : int or float, optional
        The maximum data size to include. Values greater than xmax are
        excluded.

    Returns
    -------
    X : array
        The sorted, unique values in the data.

    probabilities : array
        The portion of the data that is less than or equal to X.
    """
    data = np.array(data)

    # If we don't have any data, just return nan
    if not data.any():
        return np.array([nan]), np.array([nan])

    data = trim_to_range(data, xmin=xmin, xmax=xmax)

    # For computing the cdf (ccdf) we need our data to be sort in
    # ascending order
    n = float(len(data))
    data = np.sort(data)

    # If there are no repeated values, then we just add probability 1/n
    # after each point, which means the cumulative distribution will just
    # be 1/n, 2/n, 3/n, ... n/n.
    all_unique = not any(data[:-1]==data[1:])
    if all_unique:
        # Note the +1; see EliasL's Issue #111
        #CDF = np.arange(1, n+1) / n
        CDF = np.arange(0, n) / n

        unique_data = data

    else:
        # If we have unique values, so points will add l/n probability,
        # where l is greater than 1, so we need to note where there are
        # duplicates.

        # This clever bit is a way of using searchsorted to rapidly
        # calculate the CDF of data with repeated values comes from Adam
        # Ginsburg's plfit code, specifically:
        # https://github.com/keflavich/plfit/commit/453edc36e4eb35f35a34b6c792a6d8c7e848d3b5#plfit/plfit.py
        #CDF = np.searchsorted(data, data, side='right') / n
        CDF = np.searchsorted(data, data, side='left') / n

        # Now we remove the duplicate points
        unique_data, unique_indices = np.unique(data, return_index=True)
        CDF = CDF[unique_indices]

    return unique_data, CDF


def ccdf(data, xmin=None, xmax=None):
    """
    Compute the complementary cumulative distribution function (CCDF) of the data.

    Parameters
    ----------
    data : list or array
        The data to compute the CCDF for.

    xmin : int or float, optional
        The minimum data size to include. Values less than xmin are excluded.

    xmax : int or float, optional
        The maximum data size to include. Values greater than xmax are
        excluded.

    Returns
    -------
    X : array
        The sorted, unique values in the data.

    probabilities : array
        The portion of the data that is greater than X.
    """
    # Calculate the cdf and take 1 - cdf
    unique_data, CDF = cdf(data, xmin=xmin, xmax=xmax)

    return unique_data, 1 - CDF


def pdf(data, xmin=None, xmax=None, linear_bins=False, bins=None):
    """
    Compute the probability density function (PDF, normalized histogram) of the
    data.

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

    Returns
    -------
    bin_edges : array
        The edges of the bins of the probability density function.

    probabilities : array
        The portion of the data that is within the bin. Length 1 less than
        bin_edges, as it corresponds to the spaces between them.
    """
    from numpy import logspace, histogram, floor, unique,asarray
    from math import ceil, log10
    data = asarray(data)
    if not xmax:
        xmax = max(data)
    if not xmin:
        xmin = min(data)

    if xmin<1:  #To compute the pdf also from the data below x=1, the data, xmax and xmin are rescaled dividing them by xmin.
        xmax2=xmax/xmin
        xmin2=1
    else:
        xmax2=xmax
        xmin2=xmin

    if bins is not None:
        bins = bins
    elif linear_bins:
        bins = range(int(xmin2), ceil(xmax2)+1)
    else:
        log_min_size = log10(xmin2)
        log_max_size = log10(xmax2)
        number_of_bins = ceil((log_max_size-log_min_size)*10)
        bins = logspace(log_min_size, log_max_size, num=number_of_bins)
        bins[:-1] = floor(bins[:-1])
        bins[-1] = ceil(bins[-1])
        bins = unique(bins)

    if xmin<1: #Needed to include also data x<1 in pdf.
        hist, edges = histogram(data/xmin, bins, density=True)
        edges=edges*xmin # transform result back to original
        hist=hist/xmin # rescale hist, so that np.sum(hist*edges)==1
    else:
        hist, edges = histogram(data, bins, density=True)

    return edges, hist


def is_discrete(data):
    """Checks if every element of the array is an integer."""
    from numpy import floor
    return (floor(data)==data.astype(float)).all()


def trim_to_range(data, xmin=None, xmax=None):
    """
    Removes elements of the data that are above xmax or below xmin (if present)

    Parameters
    ----------
    data : list or array
        The data to trim.

    xmin : int or float, optional
        The minimum data size to include. Values less than xmin are excluded.

    xmax : int or float, optional
        The maximum data size to include. Values greater than xmax are
        excluded.

    Returns
    -------
    trimmed_data : array
        Trimmed data.
    """
    data = np.asarray(data)

    if xmin:
        data = data[data >= xmin]

    if xmax:
        data = data[data <= xmax]

    return data

