# Functions for the Yule distribution, conditional on  being in the right/upper
# tail, and estimation from data

# Requires zeta.R

### Function for fitting to right/upper tail of tail:
# yule.fit		Fit Yule to tail of data by numerical likelihood
#			maximization
### Distributional functions, per R standards:
# dyule			Probability mass function
# pyule			Cumulative probability function
### Backstage function, not intended for users:
# yule.loglike		Calculate log likelihood

# Probability mass function of the Yule distribution (right-tail-conditional)
# Input: data vector, distributional parameter, lower threshold, log flag
# Output: Vector of (log) probabilities
dyule <- function(x, alpha, xmin=1, log=FALSE) {
  if (xmin==1) {
    if (log) {C <- 0} else {C <- 1}
  } else { 
      C <- pyule(xmin-1,alpha,xmin=1,log.p=log,lower.tail=FALSE)
  }
  g <- function(x) {log(alpha-1) + lbeta(x,alpha) - C}
  if (log) {
   f <- function(x) {log(alpha-1) + lbeta(x,alpha) - C }
  } else {
   f <- function(x) {(alpha-1)*beta(x,alpha)/C}
  }
  d <- ifelse(x < xmin, NA, f(x))
  return(d)
}

# Cumulative distribution function of the Yule distribution
# (right-tail-conditional)
# If the threshold xmin > 1, then it calls itself recursively, reducing to the
# xmin==1 base case in one step
# Input: data vector, distributional parameter, usual flags
# Output: vector of (log) probabilities
pyule <- function(x, alpha, xmin=1, log.p=FALSE, lower.tail=TRUE) {
  if (xmin==1) {
    if (log.p) {C <- 0} else {C <- 1}
  } else {
    C <- pyule(xmin,alpha,xmin=1,log.p=log.p,lower.tail=FALSE)
  }
  g <- function(x) {x*beta(x,alpha)/C }
  g.log <- function(x) {log(x) + lbeta(x,alpha)-C }
  if (!lower.tail && log.p) { f <- function(x) { g.log(x) } }
  if (!lower.tail && !log.p) { f <- function(x) { g(x) } }
  if (lower.tail && log.p) { f<-function(x) { log(1-g(x)) } }
  if (lower.tail && !log.p) { f<-function(x) { 1-g(x) } }
  p <- ifelse(x < xmin, NA, f(x))
  return(p)
}

# Log-likelihood function of the Yule distribution
# Input: Data vector, distributional parameter, lower threshold
# Output: real-valued log-likelihood
yule.loglike <- function(x, alpha,xmin) {
  sum(dyule(x,alpha,xmin,log=TRUE))
}

# Fit Yule distribution by maximum likelihood
# Numerical minimization of the negative log-likelihood function, using the
# estimator of the zeta distribution to get an initial value for the parameter
# Input: Data vector, lower threshold
# Output: List giving the distribution type ("Yule"), the parameter, and some
#         information about the fit
# Requires: zeta.R
yule.fit <- function(x,xmin) {
  x <- x[x>=xmin]
  n <- length(x)
  negloglike <- function(a) { -yule.loglike(x,a,xmin) }
  # Invoke zeta estimator, in simplified discrete form, to get an initial
  # value
  a0 <- zeta.fit(x,threshold=xmin,method="ml.approx")$exponent
  est <- nlm(f=negloglike,p=a0)
  fit <- list(type="Yule", threshold=xmin, exponent=est$estimate[1],
              loglike=-est$minimum, samples.over.threshold=n)
  return(fit)
}