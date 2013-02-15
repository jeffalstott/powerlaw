# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import powerlaw

# <codecell>

word = genfromtxt('words.txt')
wordfit = powerlaw.Fit(word, discrete=True)

# <codecell>

bl = genfromtxt('blackouts.txt')
blfit = powerlaw.Fit(bl, discrete=False)

# <codecell>

%load_ext rmagic 

# <codecell>

xmin = blfit.xmin

# <codecell>

%%R -i bl,xmin -o shalizi_rate
source('pli-R-v0.0.3-2007-07-25/exp.R')
blfit <- exp.fit(bl, threshold=xmin)
shalizi_rate = blfit$rate
print(blfit)

# <codecell>

r = blfit.exponential.Lambda
print r, shalizi_rate

# <codecell>

%%R -i r,xmin,bl -o shalizi_like
shalizi_like=exp.loglike.tail(bl,r,xmin)

# <codecell>

alstott_like = log(powerlaw.exponential_likelihoods(bl, r, xmin))

# <codecell>

print sum(alstott_like), shalizi_like

# <codecell>

%%R -i bl,xmin -o meanlog,sdlog
source('pli-R-v0.0.3-2007-07-25/lnorm.R')
blfit <- lnorm.fit(bl, threshold=xmin)
meanlog <- blfit$meanlog
sdlog <- blfit$sdlog
print(blfit)

# <codecell>

print blfit.lognormal.parameter1, blfit.lognormal.parameter2

# <codecell>

alstott_like = log(powerlaw.lognormal_likelihoods(bl, meanlog[0], sdlog[0], xmin))
print alstott_like

# <codecell>

print blfit.power_law.alpha
print blfit.loglikelihood_ratio('power_law', 'lognormal', normalized_ratio=True)
print blfit.loglikelihood_ratio('power_law', 'exponential', normalized_ratio=True)
print blfit.loglikelihood_ratio('power_law', 'stretched_exponential', normalized_ratio=True)
print blfit.loglikelihood_ratio('power_law', 'truncated_power_law')

# <codecell>

print wordfit.power_law.alpha
print wordfit.loglikelihood_ratio('power_law', 'lognormal', normalized_ratio=True)
print wordfit.loglikelihood_ratio('power_law', 'exponential', normalized_ratio=True)
print wordfit.loglikelihood_ratio('power_law', 'stretched_exponential', normalized_ratio=True)
print wordfit.loglikelihood_ratio('power_law', 'truncated_power_law')

