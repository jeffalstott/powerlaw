# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

# <codecell>

import pylab
pylab.rcParams['xtick.major.pad']='8'
pylab.rcParams['ytick.major.pad']='8'
#import matplotlib.gridspec as gridspec
from matplotlib import rc
rc('text', usetex=False)
rc('font', family='serif')

# <codecell>

from os import listdir
files = listdir('.')
if 'blackouts.txt' not in files:
    import urllib
    urllib.urlretrieve('https://raw.github.com/jeffalstott/powerlaw/master/manuscript/blackouts.txt', 'blackouts.txt')
if 'words.txt' not in files:
    import urllib
    urllib.urlretrieve('https://raw.github.com/jeffalstott/powerlaw/master/manuscript/words.txt', 'words.txt')
if 'worm.txt' not in files:
    import urllib
    urllib.urlretrieve('https://raw.github.com/jeffalstott/powerlaw/master/manuscript/worm.txt', 'worm.txt')

# <codecell>

from numpy import genfromtxt
blackouts = genfromtxt('blackouts.txt')#/10**3
words = genfromtxt('words.txt')
worm = genfromtxt('worm.txt')
worm = worm[worm>0]

# <codecell>

def plot_basics(data, data_inst, fig, units):
    from powerlaw import plot_pdf, Fit, pdf
    annotate_coord = (-.4, .95)
    ax1 = fig.add_subplot(n_graphs,n_data,data_inst)
    plot_pdf(data[data>0], ax=ax1, linear_bins=True, color='r', linewidth=.5)
    x, y = pdf(data, linear_bins=True)
    ind = y>0
    y = y[ind]
    x = x[:-1]
    x = x[ind]
    ax1.scatter(x, y, color='r', s=.5)
    plot_pdf(data[data>0], ax=ax1, color='b', linewidth=2)
    from pylab import setp
    setp( ax1.get_xticklabels(), visible=False)
    #ax1.set_xticks(ax1.get_xticks()[::2])
    ax1.set_yticks(ax1.get_yticks()[::2])
    locs,labels = yticks()
    #yticks(locs, map(lambda x: "%.0f" % x, log10(locs)))
    if data_inst==1:
        ax1.annotate("A", annotate_coord, xycoords="axes fraction", fontsize=14)

    
    from mpl_toolkits.axes_grid.inset_locator import inset_axes
    ax1in = inset_axes(ax1, width = "30%", height = "30%", loc=3)
    ax1in.hist(data, normed=True, color='b')
    ax1in.set_xticks([])
    ax1in.set_yticks([])

    
    ax2 = fig.add_subplot(n_graphs,n_data,n_data+data_inst, sharex=ax1)
    plot_pdf(data, ax=ax2, color='b', linewidth=2)
    fit = Fit(data, xmin=1, discrete=True)
    fit.power_law.plot_pdf(ax=ax2, linestyle=':', color='g')
    p = fit.power_law.pdf()
    #ax2.set_ylim(min(p), max(p))
    ax2.set_xlim(ax1.get_xlim())
    
    fit = Fit(data, discrete=True)
    fit.power_law.plot_pdf(ax=ax2, linestyle='--', color='g')
    from pylab import setp
    setp( ax2.get_xticklabels(), visible=False)
    #ax2.set_xticks(ax2.get_xticks()[::2])
    if ax2.get_ylim()[1] >1:
        ax2.set_ylim(ax2.get_ylim()[0], 1)
    
    ax2.set_yticks(ax2.get_yticks()[::2])
    #locs,labels = yticks()
    #yticks(locs, map(lambda x: "%.0f" % x, log10(locs)))
    if data_inst==1:
       ax2.annotate("B", annotate_coord, xycoords="axes fraction", fontsize=14)        
       ax2.set_ylabel(r"$p(X)$")# (10^n)")
        
    ax3 = fig.add_subplot(n_graphs,n_data,n_data*2+data_inst)#, sharex=ax1)#, sharey=ax2)
    fit.power_law.plot_pdf(ax=ax3, linestyle='--', color='g')
    fit.exponential.plot_pdf(ax=ax3, linestyle='--', color='r')
    fit.plot_pdf(ax=ax3, color='b', linewidth=2)
    
    #p = fit.power_law.pdf()
    ax3.set_ylim(ax2.get_ylim())
    ax3.set_yticks(ax3.get_yticks()[::2])
    ax3.set_xlim(ax1.get_xlim())
    
    #locs,labels = yticks()
    #yticks(locs, map(lambda x: "%.0f" % x, log10(locs)))
    if data_inst==1:
        ax3.annotate("C", annotate_coord, xycoords="axes fraction", fontsize=14)

    #if ax2.get_xlim()!=ax3.get_xlim():
    #    zoom_effect01(ax2, ax3, ax3.get_xlim()[0], ax3.get_xlim()[1])
    ax3.set_xlabel(units)

# <codecell>

n_data = 3
n_graphs = 4
f = figure(figsize=(8,11))

data = words
data_inst = 1
units = 'Word Frequency'
plot_basics(data, data_inst, f, units)

data_inst = 2
#data = city
#units = 'City Population'
data = worm
units = 'Neuron Connections'
plot_basics(data, data_inst, f, units)

data = blackouts
data_inst = 3
units = 'Population Affected\nby Blackouts'
plot_basics(data, data_inst, f, units)

f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.3, hspace=.2)
f.savefig('FigWorkflow.eps', bbox_inches='tight')

# <codecell>

blackouts = blackouts/10**3

# <codecell>

data = blackouts
####
import powerlaw
fit = powerlaw.Fit(data)
fit.power_law.alpha
fit.power_law.sigma
fit.distribution_compare('power_law', 'exponential')

# <codecell>

data = words
####
figPDF = powerlaw.plot_pdf(data, color='b')
powerlaw.plot_pdf(data, linear_bins=True, color='r', ax=figPDF)
####
figPDF.set_ylabel(r"$p(X)$")
figPDF.set_xlabel(r"Word Frequency")
savefig('FigPDF.eps', bbox_inches='tight')

# <codecell>

data = words
fit = powerlaw.Fit(data, discrete=True)
####
figCCDF = fit.plot_pdf(color='b', linewidth=2)
fit.power_law.plot_pdf(color='b', linestyle='--', ax=figCCDF)
fit.plot_ccdf(color='r', linewidth=2, ax=figCCDF)
fit.power_law.plot_ccdf(color='r', linestyle='--', ax=figCCDF)
####
figCCDF.set_ylabel(r"$p(X)$,  $p(X\geq x)$")
figCCDF.set_xlabel(r"Word Frequency")
savefig('FigCCDF.eps', bbox_inches='tight')

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
import powerlaw
fit = powerlaw.Fit(data)
fit.xmin
fit.fixed_xmin
fit.alpha
fit.D
fit = powerlaw.Fit(data, xmin=1.0)
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
fit = powerlaw.Fit(data, xmax=10000.0)
fit.xmax
fit.fixed_xmax

# <codecell>

data = words
#FigCCDFmax = powerlaw.plot_ccdf(data, linewidth=3)
fit = powerlaw.Fit(data, discrete=True, xmax=None)
FigCCDFmax = fit.plot_ccdf(color='b', label=r"Empirical, no $x_{max}$")
fit.power_law.plot_ccdf(color='b', linestyle='--', ax=FigCCDFmax, label=r"Fit, no $x_{max}$")
fit = powerlaw.Fit(data, discrete=True, xmax=1000)
fit.plot_ccdf(color='r', label=r"Empirical, $x_{max}=1000$")
fit.power_law.plot_ccdf(color='r', linestyle='--', ax=FigCCDFmax, label=r"Fit, $x_{max}=1000$")
#x, y = powerlaw.ccdf(data, xmax=max(data))
#fig1.plot(x,y)
####
FigCCDFmax.set_ylabel(r"$p(X\geq x)$")
FigCCDFmax.set_xlabel(r"Word Frequency")
handles, labels = FigCCDFmax.get_legend_handles_labels()
leg = FigCCDFmax.legend(handles, labels, loc=3)
leg.draw_frame(False)
savefig('FigCCDFmax.eps', bbox_inches='tight')

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

data = worm
fit = powerlaw.Fit(data, discrete=True)
####
fit.distribution_compare('power_law', 'exponential')
fit.distribution_compare('power_law', 'truncated_power_law')

# <codecell>

data = words
fit = powerlaw.Fit(data, discrete=True)
####
fit.distribution_compare('power_law', 'lognormal')
fig = fit.plot_ccdf(linewidth=3, label='Empirical Data')
fit.power_law.plot_ccdf(ax=fig, color='r', linestyle='--', label='Power law fit')
fit.lognormal.plot_ccdf(ax=fig, color='g', linestyle='--', label='Lognormal fit')
####
fig.set_ylabel(r"$p(X\geq x)$")
fig.set_xlabel(r"Word Frequency")
handles, labels = fig.get_legend_handles_labels()
fig.legend(handles, labels, loc=3)
savefig('FigLognormal.eps', bbox_inches='tight')

# <codecell>

data = blackouts
fit = powerlaw.Fit(data)
####
fit.loglikelihood_ratio('power_law', 'truncated_power_law')
fit.loglikelihood_ratio('exponential', 'stretched_exponential')

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
fit = powerlaw.Fit(data, sigma_threshold=.1)
print fit.xmin, fit.D, fit.alpha
fit = powerlaw.Fit(data)
print fit.xmin, fit.D, fit.alpha
####
from matplotlib.pylab import plot
plot(fit.xmins, fit.Ds, label=r'$D$')
plot(fit.xmins, fit.sigmas, label=r'$\sigma$', linestyle='--')
plot(fit.xmins, fit.sigmas/fit.alphas, label=r'$\sigma /\alpha$', linestyle='--')
####
ylim(0, .4)
legend(loc=4)
xlabel(r'$x_{min}$')
ylabel(r'$D,\sigma,\alpha$')
savefig('FigD.eps', bbox_inches='tight')

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

