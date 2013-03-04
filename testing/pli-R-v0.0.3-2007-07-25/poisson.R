# Poisson distribution for the right/uppper tail, and its estimation

# R by default provides dpois (probability mass function), ppois (cumulative
# probability function), qpois (quantile function) and rpois (random variate
# generation)

### Function for fitting:
# pois.tail.fit		Fit Poisson to tail of data by numerically maximizing
#			likelihood
### Distributional function, per R standard:
# dpois.tail		Probability mass function, conditional on being in tail
### Backstage function, not intended for users:
# pois.tail.loglike	Calcualte log-likelihood

# Probability mass function, conditional on being in tail
# Outputs "NA" on data points below cut-off
# Input: Data vector, Poisson rate, lower cut-off, log-flag
# Output: Vector of (log) probabilities
dpois.tail <- function(x, rate, threshold=0, log=FALSE) {
  C <- ppois(threshold-1,rate,lower.tail=FALSE,log.p=log)
  if (log) {
    f <- function(x) {-C + dpois(x,rate,log=TRUE)}
  } else {
    f <- function(x) {dpois(x,rate)/C}
  }
  d <- ifelse(x<threshold, NA, f(x))
  return(d)
}

# Calculate tail-conditional log likelihood
# Input: Data vector, Poisson rate, lower cut-off
# Output: Log likelihood
pois.tail.loglike <- function(x, rate, threshold=0) {
  n <- length(x)
  JointProb <- sum(dpois(x,rate,log=TRUE))
  ProbOverThreshold <- ppois(threshold-1,rate,lower.tail=FALSE,log.p=TRUE)
  return(JointProb - n*ProbOverThreshold)
}

# Fit Poisson distribution to right/upper tail of data, by numerically
# maximizing log likelihood
# Input: Data vector, lower cut-off
# Output: List giving distribution type, estimated rate, information about
#	  fit and data
pois.tail.fit <- function(x, threshold=0) {
  # Do a simple moment-based fit first on the whole data as a pump-primer
  rate_0 <- mean(x)
  # discard data points below threshold
  x <- x[x>=threshold]
  rate_1 <- mean(x)
  n <- length(x)
  negloglike <- function(rate) { -pois.tail.loglike(x, rate, threshold) }
  est <- nlm(f=negloglike, p=rate_0)
  fit <- list(type="pois.tail", rate = est$estimate[1], threshold=threshold,
              loglike=-est$minimum, samples.over.threshold=n, full.mean=rate_0,
              mean.over.threshold=rate_1)
  return(fit)
}