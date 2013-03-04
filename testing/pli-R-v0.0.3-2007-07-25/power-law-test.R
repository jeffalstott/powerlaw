# Code for testing fits to power-law distributions against other heavy-tailed
# alternatives

# Code for ESTIMATING the heavy-tailed alternatives live in separate files,
# which you'll need to load.

# Tests are of two sorts, depending on whether or not the power law is embeded
# within a strictly larger class of distributions ("nested models"), or the two
# alternatives do not intersect ("non-nested").  In both cases, our procedures
# are based on those of Vuong (1989), neglecting his third case where the two
# hypothesis classes intersect at a point, which happens to be the best-fitting
# hypothesis.

# In the nested case, the null hypothesis is that the best-fitting distribution
# lies in the smaller class.  Then 2* the log of the likelihood ratio should
# (asymptotically) have a chi-squared distribution.  This is almost the
# classical Wilks (1938) situation, except that Vuong shows the same results
# hold even when all the distributions are "mis-specified" (hence we speak of
# best-fitting distribution, not true distribution).

# In the non-nested case, the null hypothesis is that both classes of
# distributions are equally far (in the Kullback-Leibler divergence/relative
# entropy sense) from the true distribution.  If this is true, the
# log-likelihood ratio should (asymptotically) have a Gaussian distribution
# with mean zero; our test statistic is the sample average of the log
# likelihood ratio, standardized by a consistent estiamte of its standard
# deviation.  If the null is false, and one class of distributions is closer to
# the truth, his test statistic goes to +-infinity with probability 1,
# indicating the better-fitting class of distributions.  (See, in particular,
# Theorem 5.1 on p. 318 of his paper.)

# Vuong, Quang H. (1989): "Likelihood Ratio Tests for Model Selection and
#	Non-Nested Hypotheses", _Econometrica_ 57: 307--333 (in JSTOR)
# Wilks, S. S. (1938): "The Large Sample Distribution of the Likelihood Ratio
#	for Testing Composite Hypotheses", _The Annals of Mathematical
#	Statistics_ 9: 60--62 (in JSTOR)

# All testing functions here are set up to work with the estimation functions
# in the accompanying files, which return some meta-data about the fitted
# distributions, including likelihoods, cut-offs, numbers of data points, etc.,
# much of which is USED by the testing routines.

### Function for nested hypothesis testing:
# power.powerexp.lrt		Test power-law distribution vs. power-law with
#				exponential cut-off
### Functions for non-nested hypothesis testing:
# vuong				Calculate mean, standard deviation, Vuong's
#				test statistic, and Gaussian p-values on
#				log-likelihood ratio vectors
# pareto.exp.llr		Makes vector of pointwise log-likelihood
#				ratio between fitted continuous power law
#				(Pareto) and exponential distributions
# pareto.lnorm.llr		Pointwise log-likelihood ratio, Pareto vs.
#				log-normal
# pareto.weibull.llr		Pointwise log-likelihood ratio, Pareto vs.
#				stretched exponential (Weibull)
# zeta.exp.llr 			Pointwise log-likelihood ratio, discrete power
#				law (zeta) vs. discrete exponential
# zeta.lnorm.llr		Pointwise log-likelihood ratio, zeta vs.
#				discretized log-normal
# zeta.poisson.llr		Pointwise log-likelihood ratio, zeta vs. Poisson
# zeta.weib.llr			Pointwise log-likelihood ratio, zeta vs. Weibull
# zeta.yule.llr			Pointwise log-likelihood ratio, zeta vs. Yule



# Test power law distribution vs. a power law with an exponential cut-off
# This is meaningful ONLY if BOTH distributions are continuous or discrete,
# and, of course, both were estimated on the SAME data set, with the SAME
# cut-off
# TODO: Check whether the distributions are comparable!
# Input: fitted power law distribution, fitted powerexp distribution
# Output: List giving log likelihood ratio and chi-squared p-value
# Recommended: pareto.R, powerexp.R, zeta.R, discpowerexp.R
power.powerexp.lrt <- function(power.d,powerexp.d) {
  lr <- (power.d$loglike - powerexp.d$loglike)
  p <- pchisq(-2*lr,df=1,lower.tail=FALSE)
  Result <- list(log.like.ratio = lr, p_value = p)
  Result
}



# Apply Vuong's test for non-nested models to vector of log-likelihood ratios
# Sample usage:
#### vuong(pareto.lnorm.llr(wealth,wealth.pareto,wealth.lnorm))
# The inner function produces a vector, giving the log of the likelihood
# ratio at every point; the outer function then reduces this and calculates
# both the normalized log likelihood ratio and the p-values.
# Input: Vector giving, for each sample point, the log likelihood ratio of
#        the two models under consideration
# Output: List giving total log likelihood ratio, mean per point, standard
#         deviation per point, Vuong's test statistic (normalized pointwise
#	  log likelihood ratio), one-sided and two-sided p-values (based on
#	  asymptotical standard Gaussian distribution)
vuong <- function(x) {
	n <- length(x)
	R <- sum(x)
	m <- mean(x)
	s <- sd(x)
	v <- sqrt(n)*m/s
	p1 <- pnorm(v)
	if (p1 < 0.5) {p2 <- 2*p1} else {p2 <- 2*(1-p1)}
	list(loglike.ratio=R,mean.LLR = m, sd.LLR = s, Vuong=v, p.one.sided=p1, p.two.sided=p2)
}


# Pointwise log-likelihood ratio between continuous power law (Pareto) and
# exponential distributions
# Input: Data vector, fitted Pareto distribution, fitted exponential
#	 distribution
# Output: Vector of pointwise log-likelihood ratios, ignoring points below the
# 	  Pareto's cut-off
# Requires: pareto.R
# Recommended: exp.R
pareto.exp.llr <- function(x,pareto.d,exp.d) {
	xmin <- pareto.d$xmin
	alpha <- pareto.d$exponent
	lambda <- exp.d$rate
	x <- x[x>=xmin]
	dpareto(x,threshold=xmin,exponent=alpha,log=TRUE) - dexp(x,lambda,log=TRUE) + pexp(xmin,lambda,lower.tail=FALSE,log.p=TRUE)
}

# Pointwise log-likelihood ratio between continuous power law (Pareto) and
# log-normal distributions
# Input: Data vector, fitted Pareto distribution, fitted lognormal distribution
# Output: Vector of pointwise log-likelihood ratios, ignoring points below the
# 	  Pareto's cut-off
# Requires: pareto.R
# Recommended: lnorm.R
pareto.lnorm.llr <- function(x,pareto.d,lnorm.d) {
        xmin <- pareto.d$xmin
        alpha <- pareto.d$exponent
        m <- lnorm.d$meanlog
        s <- lnorm.d$sdlog
	x <- x[x>=xmin]
	dpareto(x,threshold=xmin,exponent=alpha,log=TRUE) - dlnorm(x,meanlog=m,sdlog=s,log=TRUE) + plnorm(xmin,meanlog=m,sdlog=s,lower.tail=FALSE,log.p=TRUE)
}

# Pointwise log-likelihood ratio between continuous power law (Pareto) and
# stretched exponential (Weibull) distributions
# Input: Data vector, fitted Pareto distribution, fitted Weibull distribution
# Output: Vector of pointwise log-likelihood ratios, ignoring points below the
# 	  Pareto's cut-off
# Requires: pareto.R
# Recommended: weibull.R
pareto.weibull.llr <- function(x,pareto.d,weibull.d) {
	xmin <- pareto.d$xmin
	alpha <- pareto.d$exponent
	shape <- weibull.d$shape
	scale <- weibull.d$scale
	x <- x[x>=xmin]
	dpareto(x,threshold=xmin,exponent=alpha,log=TRUE) - dweibull(x,shape=shape,scale=scale,log=TRUE) + pweibull(xmin,shape=shape,scale=scale,lower.tail=FALSE,log.p=TRUE)
}

# Pointwise log-likelihood ratio between discrete power law (zeta) and discrete
# exponential distributions
# Input: Data vector, fitted zeta distribution, fitted discrete exponential
#	 distribution
# Output: Vector of pointwise log-likelihood ratios, ignoring points below the
# 	  zeta's cut-off
# Requires: zeta.R, exp.R
zeta.exp.llr <- function(x,zeta.d,exp.d) {
  xmin <- zeta.d$threshold
  alpha <- zeta.d$exponent
  lambda <- exp.d$lambda
  x <- x[x>=xmin]
  dzeta(x,xmin,alpha,log=TRUE) - ddiscexp(x,lambda,xmin,log=TRUE)
}

# Pointwise log-likelihood ratio between discrete power law (zeta) and discrete
# log-normal distributions
# Input: Data vector, fitted zeta distribution, fitted discrete log-nromal
#	 distribution
# Output: Vector of pointwise log-likelihood ratios, ignoring points below the
# 	  zeta's cut-off
# Requires: zeta.R, disclnorm.R
zeta.lnorm.llr <- function(x,zeta.d,lnorm.d) {
  xmin <- zeta.d$threshold
  alpha <- zeta.d$exponent
  meanlog <- lnorm.d$meanlog
  sdlog <- lnorm.d$sdlog
  x <- x[x>=xmin]
  dzeta(x,xmin,alpha,log=TRUE) - dlnorm.tail.disc(x,meanlog,sdlog,xmin,log=TRUE)
}

# Pointwise log-likelihood ratio between discrete power law (zeta) and Poisson
# distributions
# Input: Data vector, fitted zeta distribution, fitted Poisson distribution
# Output: Vector of pointwise log-likelihood ratios, ignoring points below the
# 	  zeta's cut-off
# Requires: zeta.R, poisson.R
zeta.poisson.llr <- function(x,zeta.d,pois.d) {
  xmin <- zeta.d$threshold
  alpha <- zeta.d$exponent
  rate <- pois.d$rate
  x <- x[x>=xmin]
  dzeta(x,threshold=xmin,exponent=alpha,log=TRUE) - dpois.tail(x,threshold=xmin,rate=rate,log=TRUE)
}

# Pointwise log-likelihood ratio between discrete power law (zeta) and discrete
# stretched exponential (Weibull) distributions
# Input: Data vector, fitted zeta distribution, fitted discrete Weibull
#	 distribution
# Output: Vector of pointwise log-likelihood ratios, ignoring points below the
# 	  zeta's cut-off
# Requires: zeta.R, discweib.R
zeta.weib.llr <- function(x,zeta.d,weib.d) {
   xmin <- zeta.d$threshold
   alpha <- zeta.d$exponent
   shape <- weib.d$shape
   scale <- weib.d$scale
   x <- x[x>=xmin]
   dzeta(x,xmin,alpha,log=TRUE) - ddiscweib(x,shape,scale,xmin,log=TRUE)
}

# Pointwise log-likelihood ratio between discrete power law (zeta) and Yule
# distributions
# Input: Data vector, fitted zeta distribution, fitted Yule distribution
# Output: Vector of pointwise log-likelihood ratios, ignoring points below the
# 	  zeta's cut-off
# Requires: zeta.R, yule.R
zeta.yule.llr <- function(x,zeta.d,yule.d) {
  xmin <- zeta.d$threshold
  alpha <- zeta.d$exponent
  beta <- yule.d$exponent
  x <- x[x>=xmin]
  dzeta(x,threshold=xmin,exponent=alpha,log=TRUE) - dyule(x,beta,xmin,log=TRUE)
}

