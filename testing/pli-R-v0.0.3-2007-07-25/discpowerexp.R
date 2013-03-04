### Discrete power-law distribution with exponential cut-off
# Revision history at end of file

# The normalizing constant of this distribution can only be obtained
# numerically.  A separate C program, contained in the file discpowerexp.C,
# which should accompany this code, does so.  This must be compiled, and the
# executable put someplace where R can run it.
# To install on a Unix system, proceed as follows:
### > gcc discpowerexp.c -O -o discpowerexp -lm
### > mv discpowerexp yourexecutablepath
# where "yourexecutablepath" is a directory where you can put executable files.
# Then edit the variable "discpowerexp.filename", below, to give the full
# path to discpowerexp.
# This is not the world's slickest installation mechanism, no.

### Function for fitting to data:
# discpowerexp.fit	Fit discrete power law with exponential cut-off to
#			right/upper tail of data (by maximum likelihood)
### Distributional functions, per R standards:
# ddiscpowerexp		Probability mass function
### Backstage functions, not intended for users:
# discpowerexp.loglike	Calculate log-likelihood
# discpowerexp.norm	Calculate normalizing constant, by invoking outside
#			routine
# discpowerexp.base	Calculate un-normalized probability mass function
# discpowerexp.log	Calculate log of un-normalized probability mass function


# Location of the external program calculating the normalizing constant
### EDIT THE FILE LOCATION TO GIVE CORRECT PATH ON YOUR SYSTEM!
# invoked by discpowerexp.norm, below
discpowerexp.filename <- "~/bin/discpowerexp"



# Probability mass function for discrete power law with exponential cut-off,
# conditional on being in the right/upper tail
# Returns NA on data points below cut-off
# Input: Data vector, distributional parameters, lower cut-off, log flag
# Output: Vector of (log) probabilities
ddiscpowerexp <- function(x,exponent,rate=0,threshold=1,log=FALSE) {
  if (rate==0) { return(dzeta(x,threshold,exponent,log=log)) }
  C <- discpowerexp.norm(threshold,exponent,rate)
  if (log) {
    f <- function(y) {discpowerexp.log(y,exponent,rate) - log(C)}
  } else {
    f <- function(y) {discpowerexp.base(y,exponent,rate)/C}
  }
  d <- ifelse(x<threshold,NA,f(x))
  return(d)
}

# Log likelihood of discrete powerexp, conditional on being in the right/upper
# tail
# Ignores data-points below cut-off
# Input: Data vector, distributional parameters, lower cut-off
# Output: Log likelihood
discpowerexp.loglike <- function(x,exponent,rate,threshold=1) {
   x <- x[x>=threshold]
   n <- length(x)
   JointProb <- sum(discpowerexp.log(x,exponent,rate))
   ProbOverThreshold <- log(discpowerexp.norm(threshold,exponent,rate))
   L <- JointProb - n*ProbOverThreshold
   return(L)
}

# Fit discrete powerlaw with exponential cut-off to right/upper tail of
# data via numerical likelihood maximization
# Optimization is constrained to make sure that parameters stay in the
# meaningful region
# Input: Data vector, lower threshold
# Output: List giving type of fitted distribution, estimated parameters,
#	  information about the data and fit
discpowerexp.fit <- function(x,threshold=1) {
  x <- x[x>=threshold]
  # Apply the MLEs for the exponential and the power-law (approx.) to
  # get starting values
  alpha_0 <- zeta.fit(x,threshold,method="ml.approx")$exponent
  lambda_0 <- discexp.fit(x,threshold)$lambda
  theta_0 <- c(alpha_0,lambda_0)
  negloglike <- function(theta) {
    -discpowerexp.loglike(x,theta[1],theta[2],threshold)
  }
  ui <- rbind(c(1,0),c(0,1))
  ci <- c(-1,0)
  est <- constrOptim(theta=theta_0,f=negloglike,grad=NULL,ui=ui,ci=ci)
  fit <- list(type="discpowerexp", exponent=est$par[1],
              rate=est$par[2], loglike = -est$value, threshold=threshold,
              samples.over.threshold=length(x))
  return(fit)
}


# Calculate normalizing constant for discrete powerexp distribution
# Input: Lower cut-off, distributional parameters
# Output: Numerical value of normalizing constant
# Requires: compiled program "discpowerexp" in appropriate location
# 	    see accompanying discpowerexp.c for this
discpowerexp.norm <- function(xmin,exponent,rate) {
  discpowerexp.command <- paste(discpowerexp.filename,exponent,rate,xmin)
  as.numeric(system(discpowerexp.command,intern=TRUE))
}

# Un-normalized powerexp probability mass function
# Input: Data vector, distributional parameters
# Output: Vector of numbers, proportional to probabilities of data points
discpowerexp.base <- function(x,exponent,rate=0) {
  x^(-exponent) * exp(-x*rate)
}

# Log of un-normalized powerexp probability mass function
# Equivalent to applying log to discpowerexp.base, but avoids some finite
# precision arithmetic in taking the log
# Input: Data vector, distributional parameters
# Output: Vector of numbers, equal to log probabilities of data points plus
#	  a proportionality constant
discpowerexp.log <- function(x,exponent,rate=0) {
  -exponent*log(x) -x*rate
}


# Revision history:
# v 0.0		2007-06-04	First release
# v 0.0.1	2007-06-29	Fixed compilation instructions to invoke math
#				library explicitly
# v 0.0.2	2007-07-25	Fixed changing EVERY instance of a variable's
#				name in loglike function, thanks to Alejandro
#				Balbin for the bug report