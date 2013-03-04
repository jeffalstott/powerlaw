# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import powerlaw
%load_ext rmagic 

# <codecell>

data = genfromtxt('words.txt')

# <codecell>

data = genfromtxt('words.txt')
####
fit = powerlaw.Fit(data, discrete=True, estimate_discrete=False)
print fit.xmin
print fit.alpha
print fit.power_law._pdf_discrete_normalizer
print fit.distribution_compare('power_law', 'lognormal')

data = genfromtxt('words.txt')*10.0**2
####
fit = powerlaw.Fit(data, discrete=True, xmin=700.0, estimate_discrete=False)
print fit.xmin
print fit.alpha
print fit.power_law._pdf_discrete_normalizer
print fit.distribution_compare('power_law', 'lognormal')

data = genfromtxt('words.txt')*10.0**1
####
fit = powerlaw.Fit(data, discrete=True, xmin=70.0, estimate_discrete=False)
print fit.xmin
print fit.alpha
print fit.power_law._pdf_discrete_normalizer
print fit.distribution_compare('power_law', 'lognormal')

# <codecell>

from scipy.special import zeta
zeta(2.0, 1.0)

# <codecell>

zeta(2.0, 10.0)/zeta(2.0, 1.0)

# <codecell>

zeta(2.0, 1.0) - zeta(2.0, 10.0)

# <codecell>

x = 0
for i in range(1, 10):
    x += i**-2 
print x

# <codecell>

from numpy import log, sum
data = genfromtxt('words.txt')
factor = 100.0
xmin = factor*7.0
data = data*factor
data = data[data>=xmin]
n = len(data)
print 1 + ( n / sum( log( data / ( xmin - 0.5) ) ))

# <codecell>

%%prun
data = genfromtxt('cities.txt')/10**3
####
import powerlaw
results = powerlaw.Fit(data, discrete=False)
results.xmin
results.fixed_xmin
results.alpha
results.D
len(results.xmins)

# <codecell>

#data = genfromtxt('flares.txt')
#discrete=False
data = genfromtxt('quakes.txt')
data = (10.0**data)/(10.0**3)
discrete=False
#data = genfromtxt('terrorism.txt')
#discrete=True
#data = genfromtxt('words.txt')
#discrete=True

# <codecell>

fit = powerlaw.Fit(data, discrete=discrete, discrete_approximation='round')#15000)
#data = fit.data
xmin = fit.xmin
print fit.xmin
print fit.alpha

# <codecell>

fit.plot_ccdf(original_data=True)
fit.D

# <codecell>

bins, CDF = fit.cdf()

# <codecell>

fit.n_tail

# <codecell>

n = 38
plot(fit.xmins[:n], fit.Ds[:n])
plot(fit.xmins[:n], fit.sigmas[:n])
plot((.794, .794), ylim())
plot((10, 10), ylim())

# <codecell>

bins, CDF = fit2.cdf()

# <codecell>

len(bins)

# <codecell>

fit2 = powerlaw.Fit(data, xmin =.794)
fit2.alpha
fit2.sigma
fit2.plot_ccdf()
fit2.D

# <codecell>

d = genfromtxt('fires.txt')
x = powerlaw.Fit(d, discrete=False)
x.sigma

# <codecell>

print fit.power_law.alpha
print fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
print fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
print fit.loglikelihood_ratio('power_law', 'stretched_exponential', normalized_ratio=True)
#print fit.loglikelihood_ratio('power_law', 'truncated_power_law')

# <codecell>

ax = fit.plot_ccdf()
fit.power_law.plot_ccdf(fit.data, ax, color='r')
fit.lognormal.plot_ccdf(fit.data, ax, color='g')
print fit.distribution_compare('power_law', 'lognormal')

# <codecell>

a_param1 = fit.exponential.parameter1
a_param2 = fit.exponential.parameter2
a_like = sum(fit.exponential.loglikelihoods(fit.data))

# <codecell>

%%R -i data,xmin,a_param1 -o s_like
source('pli-R-v0.0.3-2007-07-25/exp.R')
#fit <- exp.fit(data, threshold=xmin)
#s_param1 <- fit$rate
s_like <- exp.loglike.tail(data,a_param1,xmin)

# <codecell>

%%R -i data,xmin,a_param1 -o s_param1,s_like
source('pli-R-v0.0.3-2007-07-25/discexp.R')
fit <- discexp.fit(data, threshold=xmin)
data <- data[data>=xmin]
s_param1 <- fit$lambda
s_like <- discexp.loglike(data,a_param1,threshold=xmin)

# <codecell>

print a_param1, s_param1
print round(s_like[0],4), round(a_like,4)
print round(s_like[0],4)==round(a_like,4)

# <codecell>

a_param1 = fit.lognormal.parameter1
a_param2 = fit.lognormal.parameter2
a_like = sum(fit.lognormal.loglikelihoods(fit.data))

# <codecell>

%%R -i data,xmin,a_param1,a_param2 -o s_param1,s_param2,s_like
source('pli-R-v0.0.3-2007-07-25/lnorm.R')
fit <- lnorm.fit(data, threshold=xmin)
s_param1 <- fit$meanlog
s_param2 <- fit$sdlog
s_like <- lnorm.loglike.tail(data,a_param1,a_param2,xmin)

# <codecell>

%%R -i data,xmin,a_param1,a_param2 -o s_param1,s_param2,s_like
source('pli-R-v0.0.3-2007-07-25/disclnorm.R')
fit <- fit.lnorm.disc(data, threshold=xmin)
s_param1 <- fit$meanlog
s_param2 <- fit$sdlog
s_like <- lnorm.tail.disc.loglike(data,a_param1,a_param2,xmin)

# <codecell>

print s_param1, s_param2, a_param1, a_param2
print round(s_like[0],4), round(a_like,4)
print round(s_like[0],4)==round(a_like,4)

# <codecell>

d = fit.data
xmin = fit.xmin

# <codecell>

%%R -i d,a_param1,a_param2 -o s_cdf
s_cdf = plnorm(d,a_param1, a_param2, lower.tail=FALSE, log.p=TRUE)

# <codecell>

lognormal = powerlaw.Lognormal(xmin=xmin, parameters=(a_param1, a_param2))
f = figure()
ax = f.add_subplot(111)
ax.scatter(d, s_cdf, color='r')
ax.scatter(d, log((1-lognormal._cdf_base_function(d))), s=1)
#ax.set_xlim(0, 2000)

# <codecell>

%%R -i d,a_param1,a_param2 -o s_pdf
s_pdf = dlnorm.tail.disc(d,a_param1, a_param2, threshold=xmin)

# <codecell>

f = figure()
ax = f.add_subplot(111)
ax.set_yscale('log')
ax.set_xscale('log')
ax.scatter(d, s_pdf, color='r')
ax.scatter(d, fit.lognormal.likelihoods(d), s=1)
f = figure()
ax = f.add_subplot(111)
ax.plot(d, s_pdf-fit.lognormal.likelihoods(d))
figure()
hist(around(s_pdf-fit.lognormal.likelihoods(d), 5), normed=True)
#ax.set_xlim(0, 2000)

