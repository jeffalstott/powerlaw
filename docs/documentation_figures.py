"""
This script will create all of the figures for the documentation.

Conventions
-----------

- All files should be saved in png format.

- Make sure to close the figures after they are saved using ``plt.close()``;
this script is not meant to display anything.

- For a typical standalone plot the default size should be 4.5x4.5 inches at
default DPI. Feel free to change this as needed, but if you have no other
constraints, try to stick to this.

"""

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import powerlaw
import os, pathlib

from tqdm import tqdm

# Just for my personal testing and logging
print(powerlaw.__version__)

# The source file directory, used for saving the images
SOURCE_DIR = os.path.join((pathlib.Path(__file__).parent.resolve()), 'source/')

# For now this will just be in the source dir, but it might be changed by
# an argument.
SAVE_DIR = os.path.join(SOURCE_DIR, 'images')

# Since we have to save lots of figures
def savefig(fname):
    plt.savefig(os.path.join(SAVE_DIR, fname), bbox_inches='tight')


if __name__ == '__main__':

    # Create the save directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f'Created save file directory: {SAVE_DIR}')
        print()


    #################################################
    # Test datasets
    #################################################

    powerlaw.plot_test_datasets()
    savefig('test_datasets.png')
    plt.close()


    #################################################
    # Visualization
    #################################################

    # 1. Simple plot of PDF, CDF, and CCDF

    data = powerlaw.load_test_dataset('words')

    powerlaw.plot_pdf(data, label='PDF')
    powerlaw.plot_cdf(data, label='CDF')
    powerlaw.plot_ccdf(data, label='CCDF')

    plt.legend()
    plt.title('words')
    plt.gcf().set_size_inches((4.5,4.5))
    savefig('words_dists.png')
    plt.close()

    # 2. Effects of binning types (log and log)
    data = powerlaw.load_test_dataset('blackouts')

    powerlaw.plot_pdf(data / 1e3, linear_bins=False, label='PDF')
    plt.legend()
    plt.title('blackouts')
    plt.gcf().set_size_inches((3.5,3.5))
    savefig('blackouts_pdf.png')
    plt.close()

    # 3. Effects of binning types (linear and linear)

    data = powerlaw.load_test_dataset('blackouts')

    powerlaw.plot_pdf(data / 1e3, linear_bins=True, label='PDF')
    plt.legend()
    plt.title('blackouts')
    plt.gcf().set_size_inches((3.5,3.5))
    savefig('blackouts_pdf_linear.png')
    plt.close()

    # 4. Effects of binning types (linear and log)
    data = powerlaw.load_test_dataset('blackouts')

    powerlaw.plot_pdf(data / 1e3, linear_bins=True, label='PDF')
    plt.legend()
    plt.title('blackouts')
    plt.xscale('log')
    plt.gcf().set_size_inches((3.5,3.5))
    savefig('blackouts_pdf_linear_log.png')
    plt.close()

    # 5. Plotting different fits

    data = powerlaw.load_test_dataset('fires')

    fit = powerlaw.Fit(data)

    # Plot the distributions based on the data
    # No need to pass the data since the Fit already has it
    fit.plot_pdf(label='PDF')

    fit.power_law.plot_pdf(linestyle='--', label='Power law fit')
    fit.exponential.plot_pdf(linestyle='--', label='Exponential fit')
    plt.title('fires')

    plt.legend()
    plt.gcf().set_size_inches((4.5,4.5))

    savefig('fires_pdf_and_fit.png')
    plt.close()


    #################################################
    # Identifying the scaling range
    #################################################

    # 1. Example of cropped PDF (fires)

    data = powerlaw.load_test_dataset('fires')

    xmin = 10
    xmax = 1e4

    fit = powerlaw.Fit(data, xmin=xmin, xmax=xmax)

    fit.plot_pdf(label='Cropped PDF')
    fit.plot_pdf(original_data=True, label='Full PDF')
    plt.title('fires')
    plt.legend()
    plt.gcf().set_size_inches((4.5,4.5))
    savefig('fires_pdf_cropped.png')
    plt.close()

    # 2. Example of cropped PDF (words)

    data = powerlaw.load_test_dataset('words')

    xmin = 10
    xmax = 1e4

    fit = powerlaw.Fit(data, xmin=xmin, xmax=xmax)

    fit.plot_pdf(label='Cropped PDF')
    fit.plot_pdf(original_data=True, label='Full PDF')
    plt.title('words')
    plt.legend()
    plt.gcf().set_size_inches((4.5,4.5))
    savefig('words_pdf_cropped.png')
    plt.close()

    # 3. Example of cropped CCDF (words)

    data = powerlaw.load_test_dataset('words')

    xmin = 10
    xmax = 1e4

    fit = powerlaw.Fit(data, xmin=xmin, xmax=xmax)

    fit.plot_ccdf(label='Cropped CCDF')
    fit.plot_ccdf(original_data=True, label='Full CCDF')
    plt.title('words')
    plt.legend()
    plt.gcf().set_size_inches((4.5,4.5))
    savefig('words_ccdf_cropped.png')
    plt.close()


    #################################################
    # Ranges and constraints
    #################################################

    # 1. Example xmin fitting without constraints

    data = powerlaw.load_test_dataset('blackouts')

    fit = powerlaw.Fit(data / 1e3)

    plt.plot(fit.xmin_fitting_results["xmins"], fit.xmin_fitting_results["distances"], label='KS distance')
    plt.plot(fit.xmin_fitting_results["xmins"], fit.xmin_fitting_results["valid_fits"], label='Is valid fit')

    plt.axvline(fit.xmin, linestyle='--', c='black', label='Optimal $x_{min}$')

    plt.xlabel('$x_{min}$')

    plt.legend()
    plt.xscale('log')
    plt.title('blackouts, no constraints')
    plt.gcf().set_size_inches((4.5,4.5))
    savefig('blackouts_xmin.png')
    plt.close()

    # 2. Example xmin fitting with constraint

    data = powerlaw.load_test_dataset('blackouts')

    def constraint(dist):
        N = 100
        return len(dist.data) - N

    constraint_dict = {"type": 'ineq',
                       "fun": constraint,
                       "dists": ['power_law']}

    fit = powerlaw.Fit(data / 1e3, parameter_constraints=constraint_dict)

    plt.plot(fit.xmin_fitting_results["xmins"], fit.xmin_fitting_results["distances"], label='KS distance')
    plt.plot(fit.xmin_fitting_results["xmins"], fit.xmin_fitting_results["valid_fits"], label='Is valid fit')

    plt.axvline(fit.xmin, linestyle='--', c='black', label='Optimal $x_{min}$')

    plt.xlabel('$x_{min}$')

    plt.legend()
    plt.xscale('log')
    plt.title('blackouts, constraint on $N$')
    plt.gcf().set_size_inches((4.5,4.5))
    savefig('blackouts_xmin_constrained.png')
    plt.close()


    #################################################
    # Continuous and discrete data
    #################################################

    # 1. Example with and without estimate discrete for small xmin

    data = powerlaw.load_test_dataset('words')

    powerlaw.plot_pdf(data[data >= 1], label='Original data')

    np.random.seed(0)
    fit = powerlaw.Fit(data, xmin=1, discrete=True, estimate_discrete=True)
    samples = fit.power_law.generate_random(size=(100000))
    samples = samples[samples <= 1e5]
    powerlaw.plot_pdf(samples, label='Sampled data\n$x_{min} = 1$, estimate_discrete=True')

    np.random.seed(0)
    fit = powerlaw.Fit(data, xmin=1, discrete=True, estimate_discrete=False)
    samples = fit.power_law.generate_random(size=(100000))
    samples = samples[samples <= 1e5]
    powerlaw.plot_pdf(samples, label='Sampled data\n$x_{min} = 1$, estimate_discrete=False')

    plt.legend()
    plt.title('words')
    plt.gcf().set_size_inches((4.5,4.5))
    savefig('words_generation_1_xmin.png')
    plt.close()

    # 2. Example with and without estimate discrete for larger xmin

    data = powerlaw.load_test_dataset('words')

    powerlaw.plot_pdf(data[data >= 10], label='Original data')

    np.random.seed(0)
    fit = powerlaw.Fit(data, xmin=10, discrete=True, estimate_discrete=True)
    samples = fit.power_law.generate_random(size=(100000))
    samples = samples[samples <= 1e5]
    powerlaw.plot_pdf(samples, label='Sampled data\n$x_{min} = 10$, estimate_discrete=True')

    np.random.seed(0)
    fit = powerlaw.Fit(data, xmin=10, discrete=True, estimate_discrete=False)
    samples = fit.power_law.generate_random(size=(100000))
    samples = samples[samples <= 1e5]
    powerlaw.plot_pdf(samples, label='Sampled data\n$x_{min} = 10$, estimate_discrete=False')

    plt.legend()
    plt.title('words')
    plt.gcf().set_size_inches((4.5,4.5))
    savefig('words_generation_10_xmin.png')
    plt.close()


    #################################################
    # Generating data
    #################################################

    # 1. Theoretical distribution

    pl = powerlaw.Power_Law(xmin=1, parameters={'alpha': 2.0})
    pl = powerlaw.Power_Law(xmin=1, parameters=[2.0])
    pl = powerlaw.Power_Law(xmin=1, alpha=2.0)

    pl.plot_pdf(label='PDF')
    pl.plot_ccdf(label='CCDF')
    plt.legend()
    plt.title('power law, $\\alpha = 2$')
    plt.gcf().set_size_inches((4.5,4.5))
    savefig('powerlaw_slope_2.png')
    plt.close()

    # 2. Generated data from theoretical distribution

    pl = powerlaw.Power_Law(xmin=1, parameters={'alpha': 2.0})
    pl = powerlaw.Power_Law(xmin=1, parameters=[2.0])
    pl = powerlaw.Power_Law(xmin=1, alpha=2.0)

    pl.plot_pdf(label='Original PDF', alpha=0.7, linewidth=2)
    samples = pl.generate_random((10000,))
    powerlaw.plot_pdf(samples, label='Sampled PDF', alpha=0.7, linewidth=2)

    plt.legend()
    plt.title('power law, $\\alpha = 2$')
    plt.gcf().set_size_inches((4.5,4.5))
    savefig('powerlaw_slope_2_sampled.png')
    plt.close()

    # 3. Generated data from theoretical distribution, unbounded

    pl = powerlaw.Power_Law(xmin=1, alpha=2.0)

    pl.plot_pdf(label='Original PDF', alpha=0.7, linewidth=2)
    samples = pl.generate_random((10000000,))
    powerlaw.plot_pdf(samples, label='Sampled PDF', alpha=0.7, linewidth=2)

    plt.legend()
    plt.title('power law, $\\alpha = 2$')
    plt.gcf().set_size_inches((4.5,4.5))
    savefig('powerlaw_slope_2_sampled_large_n.png')
    plt.close()

    # 4. Generated data from theoretical distribution, bounded

    pl = powerlaw.Power_Law(xmin=1, xmax=1e5, alpha=2.0)

    pl.plot_pdf(label='Original PDF', alpha=0.7, linewidth=2)
    samples = pl.generate_random((10000000,))
    powerlaw.plot_pdf(samples, label='Sampled PDF', alpha=0.7, linewidth=2)

    plt.legend()
    plt.title('power law, $\\alpha = 2$')
    plt.gcf().set_size_inches((4.5,4.5))
    savefig('powerlaw_slope_2_sampled_bounded.png')
    plt.close()

    # 5. Generated data from sampled distribution, unbounded

    data = powerlaw.load_test_dataset('flares')

    fit = powerlaw.Fit(data)

    fit.plot_pdf(label='Original PDF', alpha=0.7, linewidth=2)
    samples = fit.power_law.generate_random((10000,))
    powerlaw.plot_pdf(samples, label='Sampled PDF', alpha=0.7, linewidth=2)

    plt.legend()
    plt.title('flares')
    plt.gcf().set_size_inches((4.5,4.5))
    savefig('flares_random_samples.png')
    plt.close()

    # 6. Generated data from sampled distribution, unbounded

    data = powerlaw.load_test_dataset('flares')

    fit = powerlaw.Fit(data, xmax=np.max(data))

    fit.plot_pdf(label='Original PDF', alpha=0.7, linewidth=2)
    samples = fit.power_law.generate_random((10000,))
    powerlaw.plot_pdf(samples, label='Sampled PDF', alpha=0.7, linewidth=2)

    plt.legend()
    plt.title('flares')
    plt.gcf().set_size_inches((4.5,4.5))
    savefig('flares_random_samples_bounded.png')
    plt.close()

    # 7. Power law fitting validation with randomly generated data

    np.random.seed(0)

    alphaArr = np.linspace(0.5, 2.5, 100)
    numSamples = 30 # per alpha value
    N = 3000

    fitAlphaArr = np.zeros((len(alphaArr), numSamples))

    for j in tqdm(range(len(alphaArr))):
        for k in range(numSamples):
            theoretical_dist = powerlaw.Power_Law(xmin=1, xmax=1e6, parameters=[alphaArr[j]])
            data = theoretical_dist.generate_random(N)
            
            fit = powerlaw.Fit(data, xmin=1, xmax=np.max(data))

            fitAlphaArr[j,k] = fit.power_law.alpha


    plt.plot(alphaArr, alphaArr, '--', c='tab:orange')
    for i in range(len(alphaArr)):
        plt.scatter(np.repeat(alphaArr[i], numSamples), fitAlphaArr[i], alpha=0.1, c='tab:blue')

    plt.xlabel('True alpha')
    plt.ylabel('Fit alpha')

    plt.title('Continuous power law validation')
    plt.gcf().set_size_inches((4.5,4.5))
    savefig('random_gen_validation_continuous.png')
    plt.close()


    #################################################
    # Advanced topics
    #################################################

    # 1. Multiple fits

    data = powerlaw.load_test_dataset('blackouts')

    fit = powerlaw.Fit(data / 1e3)

    plt.plot(fit.xmin_fitting_results["xmins"], fit.xmin_fitting_results["distances"], label='KS distance')
    plt.plot(fit.xmin_fitting_results["xmins"], fit.xmin_fitting_results["valid_fits"], label='Is valid fit')
    plt.plot(fit.xmin_fitting_results["xmins"], fit.xmin_fitting_results["alpha"], label='Alpha value')

    plt.xlabel('$x_{min}$')
    plt.ylabel('Distance (KS)')

    plt.legend()
    plt.xscale('log')
    plt.title('blackouts')
    plt.gcf().set_size_inches((4.5,4.5))
    savefig('blackouts_multiple_fits.png')
    plt.close()

    # 2. Multiple fits, with constraint

    data = powerlaw.load_test_dataset('blackouts')

    fit = powerlaw.Fit(data / 1e3, parameter_ranges={"alpha": [2, 3]})

    plt.plot(fit.xmin_fitting_results["xmins"], fit.xmin_fitting_results["distances"], label='KS distance')
    plt.plot(fit.xmin_fitting_results["xmins"], fit.xmin_fitting_results["valid_fits"], label='Is valid fit')
    plt.plot(fit.xmin_fitting_results["xmins"], fit.xmin_fitting_results["alpha"], label='Alpha value')

    plt.xlabel('$x_{min}$')
    plt.ylabel('Distance (KS)')

    plt.legend()
    plt.xscale('log')
    plt.title('blackouts')
    plt.gcf().set_size_inches((4.5,4.5))
    savefig('blackouts_multiple_fits_bounded.png')
    plt.close()

