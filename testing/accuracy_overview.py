"""
This script will generate several figures that give an overview of how the
library performs at fitting various distributions.
"""

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import itertools
import powerlaw

# Big font and nice DPI
plt.rcParams["font.size"] = 14
plt.rcParams["figure.dpi"] = 150

FIGURE_OUTPUT_NAME = 'accuracy_overview.png'


def randomPowerLaw(alpha, xmin, xmax, size=1):
    """
    Power-law gen for:
        pdf(x) ~ x^{alpha} for xmin <= x <= xmax

    Note that this form is slightly different than that of 
    ``powerlaw.Power_Law._generate_random_continuous()``.
    """
    r = np.random.random(size=size)
    xming, xmaxg = xmin**(alpha+1), xmax**(alpha+1)
    return (xming + (xmaxg - xming)*r)**(1./(alpha+1))


def randomPowerLawXmin(alpha, x0, xmax, size=1):
    """
    Power-law gen for:
        pdf(x) ~ const       for x0   <= x <= xmin
        pdf(x) ~ x^{alpha}   for xmin <= x <= xmax

    ie. generate a flat distribution that then decays like a power law
    so we can test the xmin fitting.

    xmin is randomly generated from a (log) uniform distribution up
    to 10**(0.1 np.log10(np.max(data))) * 0.01 is chosen so that
    way we always have at least two decades to fit the powerlaw to.
    """
    maxExp = np.log10(xmax) - 2
    xmin = 10**np.random.uniform(np.log10(x0), maxExp)
    nonPowerLawFrac = (xmin - x0) / (xmax - x0)

    r = np.random.random(size=int(size*(1 - nonPowerLawFrac)))
    xming, xmaxg = xmin**(alpha+1), xmax**(alpha+1)
    powerLawValues = (xming + (xmaxg - xming)*r)**(1./(alpha+1))
    hist, bins = np.histogram(powerLawValues, bins=np.logspace(np.log10(xmin), np.log10(xmax), 10))

    nonPowerLawSamples = (xmin - x0) / (bins[1] - bins[0]) * hist[0]
    nonPowerLawValues = np.random.uniform(x0, xmin, size=int(nonPowerLawSamples))

    return np.concatenate((nonPowerLawValues, powerLawValues)), xmin


def powerlaw_benchmark(ax=None):
    """
    Test exponent fitting for power law distributed data.
    """

    if not ax:
        fig, ax = plt.subplots()

    alpha_arr = np.linspace(0.5, 2.5, 100)
    num_samples = 30
    N = 3000

    fit_alpha_arr = np.zeros((len(alpha_arr), num_samples))

    for i in tqdm(range(len(alpha_arr))):
        for j in range(num_samples):
            data = randomPowerLaw(-alpha_arr[i], xmin=1, xmax=1e6, size=N)
            fit = powerlaw.Fit(data=data, xmin=1, xmax=np.max(data), verbose=0)

            fit_alpha_arr[i,j] = fit.power_law.alpha

    ax.plot(alpha_arr, alpha_arr, '--', c='tab:orange')
    for i in range(len(alpha_arr)):
        ax.scatter(np.repeat(alpha_arr[i], num_samples), fit_alpha_arr[i], alpha=0.1, c='tab:blue')

    ax.set_xlabel('True alpha')
    ax.set_ylabel('Fit alpha')
    ax.set_title('Basic Power Law Fitting')

    return ax


def powerlaw_xmin_benchmark(ax=None):
    """
    Test exponent and xmin fitting for data distributed piecewise: a constant
    value in the range [x0, xmin], and power law distributed in the range
    [xmin, xmax].

    Takes much longer than powerlaw_benchmark because of the xmin fitting.
    """

    if not hasattr(ax, '__iter__') and not ax:
        fig, ax = plt.subplots(1, 2, figsize=(7,3.5))

    np.random.seed(0)

    alpha_arr = np.linspace(0.5, 2.5, 25)
    num_samples = 10
    N = 3000

    # For distribution with no constant regime
    fit_alpha_arr_pure = np.zeros((len(alpha_arr), num_samples))

    # For distribution with constant regime
    xmin_arr = np.zeros((len(alpha_arr), num_samples))
    fit_alpha_arr = np.zeros((len(alpha_arr), num_samples))
    fit_xmin_arr = np.zeros((len(alpha_arr), num_samples))

    for i in tqdm(range(len(alpha_arr))):
        for j in range(num_samples):
            data = randomPowerLaw(-alpha_arr[i], xmin=1, xmax=1e6, size=N)
            fit = powerlaw.Fit(data=data, xmin=1, xmax=np.max(data), verbose=0)

            fit_alpha_arr_pure[i,j] = fit.power_law.alpha


            data, xmin = randomPowerLawXmin(-alpha_arr[i], x0=1, xmax=1e6, size=N)
            fit = powerlaw.Fit(data, xmax=np.max(data), fit_method='KS', verbose=0)

            xmin_arr[i,j] = xmin
            fit_alpha_arr[i,j] = fit.power_law.alpha
            fit_xmin_arr[i,j] = fit.xmin


    repeated_alpha_arr = np.array([np.repeat(alpha_arr[i], num_samples) for i in range(len(alpha_arr))]).flatten()
    ax[0].plot(alpha_arr, alpha_arr, '--', c='black')
    ax[0].scatter(repeated_alpha_arr, fit_alpha_arr_pure.flatten(), alpha=0.3, c='tab:blue', label='Pure power law')
    ax[0].scatter(repeated_alpha_arr, fit_alpha_arr.flatten(), alpha=0.3, c='tab:orange', label='With xmin fitting')

    ax[0].set_xlabel('True $\\alpha$')
    ax[0].set_ylabel('Fit $\\hat \\alpha$')
    ax[0].set_ylim([alpha_arr[0]-0.5, alpha_arr[-1]+0.5])
    ax[0].legend()
    ax[0].set_title('$\\alpha$ Fitting for Power Laws')

    ax[1].plot(np.linspace(np.min(xmin_arr), np.max(xmin_arr), 2), np.linspace(np.min(xmin_arr), np.max(xmin_arr), 2), '--', c='black')
    ax[1].scatter(xmin_arr.flatten(), fit_xmin_arr.flatten(), alpha=0.3, c='tab:orange')

    ax[1].set_xlabel('True $x_{min}$')
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Fit $\\hat x_{min}$')
    ax[1].set_xscale('log')
    ax[1].set_title('$x_{min}$ Fitting for Power Laws')
    #fig.suptitle('Fitting metrics for power law distributions\npowerlaw v1.6.0')

    return ax


def distribution_benchmark(parameters_list,
                           distribution,
                           num_samples=5,
                           xrange=[1, 1e6],
                           discrete=False):
    """
    Generate data from a distribution based on given parameters and fit the
    random data.

    This is the central method for all of the distribution benchmarking
    methods (except power law). Those methods are essentially just plotting
    functions for this one, specific to each distribution.
    """
    np.random.seed(0)

    N = 3000 # number of random numbers to generate and fit

    num_params = len(parameters_list[0])

    # We set the last dimension to size 3 instead of num_params because
    # all distributions have 3 parameters in this version of powerlaw, some
    # are just None.
    fit_parameters_arr = np.zeros((len(parameters_list), num_samples, 3))
    distribution_name = str(distribution().name)
    
    # TODO Use this in the new version (PR #115)
    #fit_parameters_arr = np.zeros((len(parameters_list), num_samples, len(distribution.parameter_names)))
    #distribution_name = distribution.name

    for i in tqdm(range(len(parameters_list)), desc=f'{distribution_name} benchmarking...'):
        for j in range(num_samples):
            # Generate data
            theoretical_dist = distribution(xmin=xrange[0], xmax=xrange[1], parameters=parameters_list[i], discrete=discrete)
            data = theoretical_dist.generate_random(N)

            if discrete:
                data = data.astype(np.int64)

            # Fit data
            fit = powerlaw.Fit(data=data, xmin=xrange[0], xmax=np.max(data), verbose=0, discrete=discrete)

            # This is a bit messy because there is no function to return all
            # the parameters of an arbitrary distribution
            fit_parameter_values = [getattr(getattr(fit, distribution_name), f'parameter{k+1}') for k in range(3)]
            fit_parameter_names = [getattr(getattr(fit, distribution_name), f'parameter{k+1}_name') for k in range(3)]
            fit_parameters_arr[i,j] = fit_parameter_values

            # TODO Use this in the new version (PR #115)
            #parameters = getattr(fit, distribution_name).parameters
            #fit_parameters_arr[i,j] = list(parameters.values())
            #fit_parameter_names = list(parameters.keys())

    # Transpose so that the first index corresponds to the parameter index
    #return fit_parameters_arr
    return np.rollaxis(fit_parameters_arr, 1).T


def exponential_benchmark(ax=None):

    if ax is None:
        fig, ax = plt.subplots()

    lambda_arr = np.logspace(-4,-2, 20)
    parameters_list = [[l] for l in lambda_arr]

    num_samples = 20

    fit_params = distribution_benchmark(parameters_list,
                           powerlaw.Exponential,
                           num_samples,
                           xrange=[1, None],
                           discrete=False)

    repeated_lambda_arr = np.array([np.repeat(lambda_arr[i], num_samples) for i in range(len(lambda_arr))]).flatten()

    ax.plot(lambda_arr, lambda_arr, '--', c='black')
    ax.scatter(repeated_lambda_arr, fit_params[0].flatten(), alpha=0.3, c='tab:green')

    mse = np.sqrt(np.mean((repeated_lambda_arr - fit_params[0].flatten())**2))

    ax.annotate(r'RMSE: $\sqrt{\langle (\lambda - \hat \lambda)^2 \rangle} = $' + f'${mse:.4}$', (0.05, 0.85), xycoords='axes fraction')

    ax.set_xscale('log')
    ax.set_xlabel('True $\\lambda$')
    ax.set_yscale('log')
    ax.set_ylabel('Fit $\\hat \\lambda$')

    ax.set_title('$\\lambda$ Fitting for Exponentials')

    return ax


def stretched_exponential_benchmark(ax=None):

    if not hasattr(ax, '__iter__') and ax is None:
        fig = plt.figure(figsize=(8.5,4))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax = [ax1, ax2]

    lambda_arr = np.logspace(-4, -2, 10)
    beta_arr = np.linspace(0.2, 0.9, 10)
    parameters_list = list(itertools.product(lambda_arr, beta_arr))

    num_samples = 10

    fit_params = distribution_benchmark(parameters_list,
                           powerlaw.Stretched_Exponential,
                           num_samples,
                           xrange=[10, None],
                           discrete=True)

    fit_params = fit_params.reshape([fit_params.shape[0], len(lambda_arr), len(beta_arr), fit_params.shape[-1]])

    true_lambda_arr = np.repeat([lambda_arr], len(beta_arr), axis=0)
    true_beta_arr = np.repeat([beta_arr], len(lambda_arr), axis=0)

    X,Y = np.meshgrid(np.log10(lambda_arr), beta_arr)

    ax[0].plot_surface(X, Y, np.log10(np.mean(fit_params[0], axis=-1).T), alpha=0.5, label=f'Fit $\\hat \\lambda$ (Mean, N={num_samples})')
    ax[0].plot_surface(X, Y, np.log10(true_lambda_arr), alpha=0.5, label='True $\\lambda$')

    ax[0].set_xlabel('True $\\lambda$ (Log)')
    ax[0].set_ylabel('True $\\beta$')
    ax[0].set_zlabel('Fit $\\hat \\lambda$ (Log)')
    ax[0].legend()

    ax[1].plot_surface(X, Y, np.mean(fit_params[1], axis=-1).T, alpha=0.5, label=f'Fit $\\hat \\beta$ (Mean, N={num_samples})')
    ax[1].plot_surface(X, Y, true_beta_arr.T, alpha=0.5, label='True $\\beta$')

    ax[1].set_xlabel('True $\\lambda$ (Log)')
    ax[1].set_ylabel('True $\\beta$')
    ax[1].set_zlabel('Fit $\\hat \\beta$')
    ax[1].legend()

    return ax


def truncated_power_law_benchmark(ax=None):

    if not hasattr(ax, '__iter__') and ax is None:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax = [ax1, ax2]

    alpha_arr = np.linspace(0.5, 2.5, 3)
    lambda_arr = np.logspace(-4, -2, 3)
    parameters_list = list(itertools.product(alpha_arr, lambda_arr))

    num_samples = 5

    fit_params = distribution_benchmark(parameters_list,
                           powerlaw.Truncated_Power_Law,
                           num_samples,
                           xrange=[10, None],
                           discrete=False)

    fit_params = fit_params.reshape([fit_params.shape[0], len(alpha_arr), len(lambda_arr), fit_params.shape[-1]])

    true_lambda_arr = np.repeat([lambda_arr], len(alpha_arr), axis=0)
    true_alpha_arr = np.repeat([alpha_arr], len(lambda_arr), axis=0)

    X,Y = np.meshgrid(alpha_arr, np.log10(lambda_arr))

    ax[0].plot_surface(X, Y, np.log10(np.mean(fit_params[0], axis=-1).T), alpha=0.5, label=f'Fit $\\hat \\alpha$ (Mean, N={num_samples})')
    ax[0].plot_surface(X, Y, np.log10(true_alpha_arr), alpha=0.5, label='True $\\alpha$')

    ax[0].set_xlabel('True $\\alpha$')
    ax[0].set_ylabel('True $\\lambda$ (Log)')
    ax[0].set_zlabel('Fit $\\hat \\alpha$')
    ax[0].legend()

    ax[1].plot_surface(X, Y, np.log10(np.mean(fit_params[1], axis=-1).T), alpha=0.5, label=f'Fit $\\hat \\lambda$ (Mean, N={num_samples})')
    ax[1].plot_surface(X, Y, np.log10(true_lambda_arr.T), alpha=0.5, label='True $\\lambda$')

    ax[1].set_xlabel('True $\\alpha$')
    ax[1].set_ylabel('True $\\lambda (Log)$')
    ax[1].set_zlabel('Fit $\\hat \\lambda$ (Log)')
    ax[1].legend()

    return ax


def lognormal_benchmark(ax=None):

    if not hasattr(ax, '__iter__') and ax is None:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax = [ax1, ax2]

    mu_arr = np.linspace(1, 10, 3)
    sigma_arr = np.linspace(1, 10, 3)
    parameters_list = list(itertools.product(mu_arr, sigma_arr))

    num_samples = 5

    fit_params = distribution_benchmark(parameters_list,
                           powerlaw.Lognormal,
                           num_samples,
                           xrange=[1, None],
                           discrete=False)

    fit_params = fit_params.reshape([fit_params.shape[0], len(mu_arr), len(sigma_arr), fit_params.shape[-1]])

    true_sigma_arr = np.repeat([sigma_arr], len(mu_arr), axis=0)
    true_mu_arr = np.repeat([mu_arr], len(sigma_arr), axis=0)

    X,Y = np.meshgrid(mu_arr, sigma_arr)

    ax[0].plot_surface(X, Y, np.mean(fit_params[0], axis=-1).T, alpha=0.5, label=f'Fit $\\hat \\mu$ (Mean, N={num_samples})')
    ax[0].plot_surface(X, Y, true_mu_arr, alpha=0.5, label='True $\\mu$')

    ax[0].set_xlabel('True $\\mu$')
    ax[0].set_ylabel('True $\\sigma$')
    ax[0].set_zlabel('Fit $\\hat \\mu$')
    ax[0].legend()

    ax[1].plot_surface(X, Y, np.mean(fit_params[1], axis=-1).T, alpha=0.5, label=f'Fit $\\hat \\sigma$ (Mean, N={num_samples})')
    ax[1].plot_surface(X, Y, true_sigma_arr.T, alpha=0.5, label='True $\\sigma$')

    ax[1].set_xlabel('True $\\mu$')
    ax[1].set_ylabel('True $\\sigma$')
    ax[1].set_zlabel('Fit $\\hat \\sigma$')
    ax[1].legend()

    return ax


if __name__ == '__main__':

    #powerlaw_xmin_benchmark()

    #exponential_benchmark()

    stretched_exponential_benchmark()

    #truncated_power_law_benchmark()

    #lognormal_benchmark()

    plt.gcf().tight_layout()

    #plt.savefig(FIGURE_OUTPUT_NAME, bbox_inches='tight')
    plt.show()
