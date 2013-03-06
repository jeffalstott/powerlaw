# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from os import listdir
files = listdir('.')
if 'blackouts.txt' not in files:
    import urllib
    urllib.urlretrieve('http://tuvalu.santafe.edu/~aaronc/powerlaws/data/blackouts.txt', 'blackouts.txt')
if 'words.txt' not in files:
    import urllib
    urllib.urlretrieve('http://tuvalu.santafe.edu/~aaronc/powerlaws/data/words.txt', 'words.txt')

# <codecell>

from numpy import genfromtxt
blackouts = genfromtxt('blackouts.txt')/10**3
words = genfromtxt('words.txt')

# <codecell>

data = blackouts
####
import powerlaw
results = powerlaw.Fit(data, discrete=True)
results.xmin
results.fixed_xmin
results.alpha
results.sigma
results.D
results = powerlaw.Fit(data, xmin=60.0)
results.xmin
results.fixed_xmin
results.alpha
results.sigma
results.D

# <codecell>

data = blackouts
####
fit = powerlaw.Fit(data, xmin=(250.0, 300.0))
fit.fixed_xmin
fit.given_xmin
fit.xmin

# <codecell>

data = blackouts
fit = powerlaw.Fit(data)
####
fit.power_law
fit.power_law.alpha
fit.power_law.parameter1
fit.power_law.parameter1_name
fit.lognormal.mu
fit.lognormal.parameter1_name
fit.lognormal.parameter2_name
fit.lognormal.parameter3_name == None

# <codecell>

data = blackouts
####
fit = powerlaw.Fit(data)
R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
print R, p

# <codecell>

data = blackouts
fit = powerlaw.Fit(data)
####
fit.loglikelihood_ratio('power_law', 'truncated_power_law')
fit.loglikelihood_ratio('exponential', 'stretched_exponential')

# <codecell>

data = blackouts
fit = powerlaw.Fit(data)
####
fit = powerlaw.Fit(data, xmin=230.0)
fit.discrete
fit = powerlaw.Fit(data, xmin=230.0, discrete=True)
fit.discrete 

# <codecell>

data = blackouts
fit = powerlaw.Fit(data)
####
fit = powerlaw.Fit(data, xmax=10000.0)
fit.xmax
fit.fixed_xmax

# <codecell>

data = words
fit = powerlaw.Fit(data, discrete=True)
####
fig1 = fit.plot_ccdf(linewidth=3, label='Empirical Data')
fit.power_law.plot_ccdf(ax=fig1, color='r', linestyle='--', label='Power law fit')
fit.lognormal.plot_ccdf(ax=fig1, color='g', linestyle='--', label='Lognormal fit')
####
fig1.set_ylabel(r"$p(X\geq x)$")
fig1.set_xlabel(r"Word Frequency")
handles, labels = fig1.get_legend_handles_labels()
fig1.legend(handles[::-1], labels[::-1], loc=3)
savefig('Fig1.eps')

# <codecell>

data = words
fit = powerlaw.Fit(data, discrete=True)
####
fig2 = fit.plot_pdf(linear_bins=True, color='r')
fit.plot_pdf(ax=fig2, color='b')
####
fig2.set_ylabel(r"$p(X)$")
fig2.set_xlabel(r"Word Frequency")
savefig('Fig2.eps')

# <codecell>

data = blackouts
fit = powerlaw.Fit(data)
###
x, y = fit.cdf()
bin_edges, probability = fit.pdf()
y = fit.lognormal.cdf(data=[300,350])
y = fit.lognormal.pdf()

# <codecell>

data = blackouts
####
results = powerlaw.Fit(data, discrete=True, estimate_discrete=True)
results.power_law.alpha
results = powerlaw.Fit(data, discrete=True, estimate_discrete=False)
results.power_law.alpha

# <codecell>

data = blackouts
####
results = powerlaw.Fit(data, discrete=True, xmin=230.0, xmax=9000, discrete_approximation='xmax')
results.lognormal.mu
results = powerlaw.Fit(data, discrete_approximation=100000, xmin=230.0, discrete=True)
results.lognormal.mu
results = powerlaw.Fit(data, discrete_approximation='round', xmin=230.0, discrete=True)
results.lognormal.mu

# <codecell>

data = blackouts
####
fit = powerlaw.Fit(data)
fit.power_law.alpha, fit.power_law.sigma, fit.xmin

fit = powerlaw.Fit(data, sigma_threshold=.1)
fit.power_law.alpha, fit.power_law.sigma, fit.xmin

parameter_range = {'alpha': [2.3, None], 'sigma': [None, .2]}
fit = powerlaw.Fit(data, parameter_range=parameter_range)
fit.power_law.alpha, fit.power_law.sigma, fit.xmin

parameter_range = lambda(self): self.sigma/self.alpha < .05
fit = powerlaw.Fit(data, parameter_range=parameter_range)
fit.power_law.alpha, fit.power_law.sigma, fit.xmin

# <codecell>

data = blackouts
####
fit = powerlaw.Fit(data, sigma_threshold=.001)
fit.power_law.alpha, fit.power_law.sigma, fit.xmin, fit.noise_flag

fit.lognormal.mu, fit.lognormal.sigma
range_dict = {'mu': [10.5, None]}
fit.lognormal.parameter_range(range_dict)
fit.lognormal.mu, fit.lognormal.sigma, fit.lognormal.noise_flag
initial_parameters = (12, .7)
fit.lognormal.parameter_range(range_dict, initial_parameters)
fit.lognormal.mu, fit.lognormal.sigma, fit.lognormal.noise_flag

# <codecell>

data = blackouts
fit = powerlaw.Fit(data, sigma_threshold=.1)
print fit.xmin, fit.D, fit.alpha
fit = powerlaw.Fit(data)
print fit.xmin, fit.D, fit.alpha
####
plot(fit.xmins, fit.Ds, label=r'$D$')
plot(fit.xmins, fit.sigmas, label=r'$\sigma$', linestyle='--')
plot(fit.xmins, fit.sigmas/fit.alphas, label=r'$\sigma /\alpha$', linestyle='--')
####
ylim(0, .4)
legend(loc=4)
xlabel(r'$x_{min}$')
ylabel(r'$D,\sigma,\alpha$')
savefig('Fig3.eps')

# <codecell>

#from sicpy.optimize import fmin_ncg
#results = powerlaw.Fit(data, fit_optimizer=fmin_ncg)
#results.power_law.alpha

