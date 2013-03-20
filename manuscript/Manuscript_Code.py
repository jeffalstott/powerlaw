# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pylab
pylab.rcParams['xtick.major.pad']='8'
pylab.rcParams['ytick.major.pad']='8'
#import matplotlib.gridspec as gridspec
#from matplotlib import rc
#rc('text', usetex=False)
#rc('font', family='serif')

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
blackouts = genfromtxt('blackouts.txt')#/10**3
words = genfromtxt('words.txt')
cities = genfromtxt('cities.txt')

# <codecell>

f = figure()
ax = f.add_subplot(111)
ax.hist(blackouts, normed=True)
#ax.ticklabel_format(style='sci', scilimits=(0,0))
ax.set_xticks([])#ax.get_xticks()[::2])
#locs,labels = yticks()
#yticks(locs, map(lambda x: "%.1f" % x, locs*1e6))

#ax.set_yticks(map(lambda x: "%.1f"%x, ax.get_yticks()*1e6))

# <codecell>


# <codecell>

def plot_basics(data, data_inst, fig, units):
    from powerlaw import plot_pdf, Fit, pdf
    ylabel_coord = -.3
    ax1 = fig.add_subplot(n_graphs,n_data,n_data+data_inst)
    plot_pdf(data, ax=ax1, linear_bins=True, color='r', linewidth=.5)
    x, y = pdf(data, linear_bins=True)
    ind = y>0
    y = y[ind]
    x = x[:-1]
    x = x[ind]
    ax1.scatter(x, y, color='r', s=.5)
    plot_pdf(data, ax=ax1, color='b', linewidth=2)
    from pylab import setp
    setp( ax1.get_xticklabels(), visible=False)
    ax1.set_yticks(ax1.get_yticks()[::2])
    locs,labels = yticks()
    yticks(locs, map(lambda x: "%.0f" % x, log10(locs)))
    if data_inst==1:
        ax1.annotate("A", (0,0.95), xycoords=(ax1.get_yaxis().get_label(), "axes fraction"), fontsize=14)

    
    from mpl_toolkits.axes_grid.inset_locator import inset_axes
    ax1in = inset_axes(ax1, width = "30%", height = "30%", loc=3)
    ax1in.hist(data, normed=True, color='b')
    ax1in.set_xticks([])
    ax1in.set_yticks([])

    
    ax = fig.add_subplot(n_graphs,n_data,n_data*2+data_inst, sharex=ax1)
    plot_pdf(data, ax=ax, color='b', linewidth=2)
    fit = Fit(data, xmin=1)
    fit.power_law.plot_pdf(ax=ax, linestyle=':', color='g')

    fit = Fit(data)
    fit.power_law.plot_pdf(ax=ax, linestyle='--', color='g')
    #from pylab import setp
    #setp( ax.get_xticklabels(), visible=False)
    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_yticks(ax.get_yticks()[::2])
    locs,labels = yticks()
    yticks(locs, map(lambda x: "%.0f" % x, log10(locs)))
    if data_inst==1:
        ax.annotate("B", (0,0.95), xycoords=(ax.get_yaxis().get_label(), "axes fraction"), fontsize=14)
        #ax.set_ylabel("P(X) (10^n)")
        
    ax = fig.add_subplot(n_graphs,n_data,n_data*3+data_inst)
    fit.plot_pdf(ax=ax, color='b', linewidth=2)
    fit.power_law.plot_pdf(ax=ax, linestyle='--', color='g')
    fit.exponential.plot_pdf(ax=ax, linestyle='--', color='r')
    p = fit.power_law.pdf()
    ax.set_ylim(min(p), max(p))
    ax.set_yticks(ax.get_yticks()[::2])
    locs,labels = yticks()
    yticks(locs, map(lambda x: "%.0f" % x, log10(locs)))
    if data_inst==1:
        ax.annotate("C", (0,0.95), xycoords=(ax.get_yaxis().get_label(), "axes fraction"), fontsize=14)
        
    ax.set_xlabel(units)

# <codecell>

n_data = 3
n_graphs = 4
f = figure(figsize=(8,11))

data = words
data_inst = 1
units = 'Word Frequency'
plot_basics(data, data_inst, f, units)

data = cities
data_inst = 2
units = 'City Population'
plot_basics(data, data_inst, f, units)

data = blackouts
data_inst = 3
units = 'Population Affected\nby Blackouts'
plot_basics(data, data_inst, f, units)

f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.2, hspace=.15)

# <codecell>

data = blackouts
####
import powerlaw
fit = powerlaw.Fit(data)
fit.power_law.alpha
fit.power_law.sigma

# <codecell>

data = blackouts
####
import powerlaw
fit = powerlaw.Fit(data)
fit.xmin
fit.fixed_xmin
fit.alpha
fit.D
fit = powerlaw.Fit(data, xmin=60.0)
fit.xmin
fit.fixed_xmin
fit.alpha
fit.D

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
fit = powerlaw.Fit(data, discrete=True, estimate_discrete=True)
fit.power_law.alpha
fit = powerlaw.Fit(data, discrete=True, estimate_discrete=False)
fit.power_law.alpha

# <codecell>

data = blackouts
####
fit = powerlaw.Fit(data, discrete=True, xmin=230.0, xmax=9000, discrete_approximation='xmax')
fit.lognormal.mu
fit = powerlaw.Fit(data, discrete_approximation=100000, xmin=230.0, discrete=True)
fit.lognormal.mu
fit = powerlaw.Fit(data, discrete_approximation='round', xmin=230.0, discrete=True)
fit.lognormal.mu

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

