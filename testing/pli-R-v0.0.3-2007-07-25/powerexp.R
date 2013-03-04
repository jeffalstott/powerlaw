########## Functions for power law with exponential cut-off
# Let's call this "powerexp" for short.
# The parameterization is 
### f(x) \propto (x/threshold)^exponent exp(-rate*x)
# Really, functions should check that rate is positive, and that
# exponent is negative (otherwise it's just gamma distribution w/ threshold)
# This is not yet implemented

# The normalization constant must be obtained numerically; it is an upper
# incomplete gamma function of negative index.  Unfortuantely, R only includes
# code for upper incomplete gamma functions of _positive_ index.  Sometimes
# the normalizing constant can be expressed in terms of the latter; if so this
# code does so.  In all other cases, the normalizing constant can be expressed
# in terms of the exponential integral function.
# A separate C program invokes the Gnu Scientific Library (GSL) to calculate
# the exponential integral.  This program should be found in an accompanying
# file called exponential-integral.tgz.
# To install it on a Unix system, first make sure GSL is installed.
### > tar xzf exponential-integral.tgz
### > cd exponential-integral
# At this point, you need to edit the file "Makefile" to give the location of
# the GSL, which on my system is /sw/lib (for the library) and /sw/include
# (for the included files).  Then
### > make
### > mv exp_int yourexecutablepath
# where "yourexecutablepath" is a directory where you can put executable
# files.  Then edit the variable "exp_int_function_filename" in this file,
# below, to give the full path to the executable program exp_int.
# This is not the world's slickest installation mechanism, no.




### R-standard functions for a distribution:
# dpowerexp		Probability density
# ppowerexp		Cumulative distribution function
# qpowerexp		Quantile function
# rpowerexp		Random variate generation
### Estimation functions:
# powerexp.fit		Find scaling exponent and exponential rate by maximizing
#			likelihood
# test.powerexp.fit	Test the quality of the powerexp.fit routine
#			by generating data from a pure Pareto and comparing
#			the result log-likelihood ratio to the chi^2
#			distribution (to which it should asymptotically
#			converge) --- mostly included for debugging of the
#			normalizing constant and numerical optimization
### Backstage functions, not for users:
# powerexp.loglike	Calculcate log-likelihood, not intended for users
# test.powerexp.fit.1	Inner loop of test.powerexp.fit (one call)
# qpowerexponce		Find one upper quantile via numerical inversion
# uigf			Upper incomplete gamma function
# exp_int		Exponential integral function (vectorized); invokes
#                       either of the two following functions
# exp_int_once.gsl	Exponential integral of one value, calling Gnu
#			Scientific Library (crudely)
# exp_int_once.nr	Exponential integral of one value, re-implementing
#			a numerical recipe in R (not included in this public
#			release for copyright reasons)


# Location of external program invoking the GSL for calculating the exponential
# integral function
exp_int_function_filename <- "~/bin/exponential_integral"
#### EDIT THIS LOCATION!!!! ###




# Density of power laws with exponential cut-off
# Returns NA on values < threshold
# Input: Data vector, lower threshold, scaling exponent, exponential rate,
#        log flag
# Output: vector of (log) densities
dpowerexp <- function(x, threshold=1, exponent, rate=0, log=FALSE) {
   # If the rate is zero, we've got a pure Pareto
   if (rate==0) {
     return(dpareto(x,threshold,exponent,log))
   }
   if (rate==0) {
     cat("I should never be here!\n")
   }
   nondim_thresh <- threshold*rate
   if (is.nan(log(nondim_thresh))) { cat("Non-dimensionalized threshold, ",nondim_thresh,", is too small for log\n")}
   if (is.nan(log(rate))) { cat("Rate, ",rate,", is too small for log\n")}
   if (is.nan(log(uigf(-exponent+1,nondim_thresh)))) {cat("UIGF, ", uigf, ", is too small for log\n")}
   prefactor <- rate/(nondim_thresh^exponent)
   C <- prefactor/uigf(-exponent+1,nondim_thresh)
   prefactor.log <- log(rate) - exponent*log(nondim_thresh)
   C.log <- prefactor.log - log(uigf(-exponent+1,nondim_thresh))
   if (is.nan(C.log)) {cat("Normalizing constant", C, ", is too small for log\n")}
   # If I want the log density, I may as well avoid some finite-precision
   # arithmetic first
   if(log) {
   	f <- function(y) {C.log - exponent*(log(y)-log(threshold)) - rate*y}
   }
   else {
	f <- function(y) {C*(y/threshold)^(-exponent)*exp(-rate*y)}
   }
   d <- ifelse(x<threshold,NA,f(x))
   return(d)
}

# Cumulative distribution function for power law with exponential cut-off
# Returns NA on values less than lower threshold
# Input: Data vector, lower threshold, scaling exponent, exponential rate,
#        usual flags
# Output: Vector of (log) probabilities
ppowerexp <- function(x, threshold=1,exponent,rate=0,
                      lower.tail=TRUE,log.p=FALSE) {
   # The complementary distribution, Pr(X > x), is very simple,
   # it's just uigf(-exponent+1,rate*x)/uigf(-exponent+1,rate*threshold)
   # So, we do that
   if (rate==0) {
     return(ppareto(x,threshold,exponent,log,lower.tail))
   }
   C <- 1/uigf(-exponent+1,rate*threshold)
   this_uigf <- function(x){uigf(-exponent+1,rate*x)}
   if ((!lower.tail)&&(!log.p)) {
     f <- function(x) {C*this_uigf(x)}
   }
   if ((lower.tail)&&(!log.p)) {
     f <- function(x) {1 - C*this_uigf(x)}
   }
   if ((!lower.tail)&&(log.p)) {
     f <- function(x) {log(C) + log(this_uigf(x))}
   }
   if ((lower.tail)&&(log.p)) {
     f <- function(x) {log(1 - c*this_uigf(x))}
   }
   p <- ifelse(x<threshold,NA,f(x))
   return(p)
}

# Quantiles of power-law with exponential cut-off distributions
# Input: Vector of (log) probabilities, lower threshold, scaling exponent,
#        exponential rate, usual flags
# Output: Vector of quantiles
qpowerexp <- function(p, threshold=1, exponent, rate=0, lower.tail=TRUE, log.p = FALSE) {
  # This isn't going to be simple, is it?  However, since ppowerexp is known,
  # numerical inversion should be possible
  if (rate==0) {
    return(qpareto(p,threshold,exponent,log.p,lower.tail))
  }
  if (log.p) {
    z <- exp(p)
  } else {
    z <- p
  }
  if (lower.tail) {
    q <- 1-z
  } else {
    q <- z
  }
  qs <- sapply(q,qpowerexponce,threshold,exponent,rate)
  return(qs)
}

# Generate random variates from a power law with exponential cut-off
# Input: Integer size, lower threshold, scaling exponent, exponential rate
# Output: Real vector of random variates
rpowerexp <- function(n,threshold=1,exponent,rate=0) {
  if (rate==0) {
    return(rpareto(n,threshold,exponent))
  }
  ru <- runif(n)
  r <- qpowerexp(ru,threshold,exponent,rate)
  return(r)
}

# Estimate scaling exponent and exponential rate of power law with
# exponential cut-off by maximizing likelihood
# Input: Data vector, lower threshold
# Output: List, giving type ("powerexp"), scaling exponent, exponential
#         rate, lower threshold, log-likelihood
powerexp.fit <- function(data,threshold=1,method="constrOptim",initial_rate=-1) {
  x <- data[data>=threshold]
  negloglike <- function(theta) {
    -powerexp.loglike(x,threshold,exponent=theta[1],rate=theta[2])
  }
  # Fit a pure power-law distribution
  pure_powerlaw <- pareto.fit(data,threshold)
  # Use this as a first guess at the exponent
  initial_exponent <- pure_powerlaw$exponent
  if (initial_rate < 0) { initial_rate <- exp.fit(data,threshold)$rate }
  minute_rate <- 1e-6
  theta_0 <- as.vector(c(initial_exponent,initial_rate))
  theta_1 <- as.vector(c(initial_exponent,minute_rate))
  switch(method,
    constrOptim = {
      # Impose the constraint that rate >= 0
      # and that exponent >= -1
      ui <- rbind(c(1,0),c(0,1))
      ci <- c(-1,0)
      # Can't start with values on the boundary of the feasible set so add
      # tiny amounts just in case
      if (theta_0[1] == -1) {theta_0[1] <- theta_0[1] + minute_rate}
      if (theta_0[2] == 0) {theta_0[2] <- theta_0[2] + minute_rate}
      est <- constrOptim(theta=theta_0,f=negloglike,grad=NULL,ui=ui,ci=ci)
      alpha <- est$par[1]
      lambda <- est$par[2]
      loglike <- -est$value},
    optim = {
      est <- optim(par=theta_0,fn=negloglike)
      alpha <- est$par[1]
      lambda <- est$par[2]
      loglike <- -est$value},
    nlm = {
      est.0 <- nlm(f=negloglike,p=theta_0)
      est.1 <- nlm(f=negloglike,p=theta_1)
      est <- est.0
      if (-est.1$minimum > -est.0$minimum) { est <- est.1;cat("NLM had to switch\n") }
      alpha <- est$estimate[1]
      lambda <- est$estimate[2]
      loglike <- -est$minimum},
    {cat("Unknown method",method,"\n"); alpha<-NA; lambda<-NA; loglike<-NA}
  )
  fit <- list(type="powerexp", exponent=alpha, rate=lambda, xmin=threshold,
              loglike=loglike, samples.over.threshold=length(x))
  return(fit)
}

# Calculate log-likelihood under power law with exponential cut-off
# Input: Data vector, lower threshold, scaling exponent, exponential rate
# Output: Real-valued log-likelihood
powerexp.loglike <- function(x,threshold=1,exponent,rate) {
  x <- x[x>=threshold]
  L <- sum(dpowerexp(x,threshold,exponent,rate,log=TRUE))
  return(L)
}

# Test the quality of the fitting by generating Pareto variates and seeing
# whether the log likelihood ratio is \chi^2
# Input: number of replicates, sample size, distributional parameters
# Output: List of 2*LLR values
# Side-effect: Plot of CDF of 2*LLR vs. pchisq(,1)
test.powerexp.fit <- function(reps=200,n=200,xmin=1,alpha=2.5) {
  l <- replicate(reps,2*test.powerexp.fit.1(n,xmin,alpha))
  plot(ecdf(l))
  curve(pchisq(x,1),add=TRUE,col="red")
  return(l)
}

test.powerexp.fit.1 <- function(n=200,xmin=1,alpha=2.5) {
  x <- rpareto(n,xmin,alpha)
  pareto.ll <- pareto.fit(x,xmin)$loglike
  powerexp.ll <- powerexp.fit(x,xmin,initial_rate=0)$loglike
  return(powerexp.ll - pareto.ll)
}


# Upper quantile of a single value from a power-law with exponential cut-off
# Should only ever be called by qpowerexp
# Input: Probability, lower threshold, scaling exponent, exponential rate
# Output: real-valued quantile
qpowerexponce <- function(q,threshold,exponent,rate) {
  # Finds a single powerexp (upper) quantile; gets called via sapplied in the
  # real qpowerexp
  # Handle easy cases first
  if (q==0) {
    return(Inf)
  }
  if (q==1) {
    return(threshold)
  }
  # The basic idea is to solve the equation
  # q = ppowerexp(x)
  # for x, by invoking the "uniroot" routine for finding zeroes of one-argument
  # functions
  w <- function(x) {q - ppowerexp(x,threshold,exponent,rate,lower.tail=FALSE)}
  # powerexp decays more quickly than a power-law distribution of the same
  # exponent and threshold
  # Use the quantile of the Pareto as an upper bound
  # Use the lower threshold as a lower bound
  pareto_x_of_q <- qpareto(q,threshold,exponent,lower.tail=FALSE)
  upperquant <- uniroot(f=w,interval=c(threshold,pareto_x_of_q))
  x <- upperquant$root
  return(x)
}




################## Numerical utility functions
# Mostly in connection with the "powerexp" distribution family


# The upper incomplete gamma function, allowing for negative indices
# Part of the normalizing constant in the powerexp family
# Uses the identity (holding for all a != 0)
### Gamma(a,x) = (1/a)(Gamma(a+1,x) - e^{-x} x^a)
# and Gamma(0,x) = E_1(x), first-order exponential integral (special) function

# Input: exponent (a), lower cut-off (x),  log flag
# Output: Real value
uigf <- function(a,x,log=FALSE) {
  if (a > 0) {
    # Use gamma-distribution tricks
    Gamma <- gamma(a)*pgamma(x,a,lower.tail=FALSE)
  }
  if (a == 0) {
    # Invoke the exponential integral function
    Gamma <- exp_int(x)
  }
  if (a < 0) {
    # Recurse
    Gamma <- (1/a) * (uigf(a+1,x) - exp(-x)*(x^a))
  }
  if (log) {
    Gamma <- log(Gamma)
  }
  return(Gamma)
}

# The exponential integral function
# Used in evaluating the upper incomplete gamma function, when the latter
# has a negative integer argument
# Two choices: invoke the GNU Scientific Library (GSL), via Cosma's
# very clumsy R/C interface
# OR use the series expansion given in Numerical Recipes, translated into
# R without any optimization or permissions
# Both methods don't work well on vector data, so exp_int acts as
# wrapper, with a choice of method, defaulting to the GSL (where my programming
# had less chance to screw it up)

# Compute the first-order exponential integrals of vectors of arguments
# The method "gsl" makes an external call to the GSL
# The method "nr" calls code lifted from Numerical Recipies
# For copyright reasons that method is NOT part of this public release
# Input: Real-valued vector, method flag
# Output: Real-valued vector
exp_int <- function(x,method="gsl") {
  switch(method,
    gsl = {ei <- sapply(x,exp_int_once.gsl)},
    nr = {ei <- sapply(x,exp_int_once.nr)},
    {cat("Unknown method", method); ei <- NA}
  )
  return(ei)
}

# Compute the exponential integral via invoking the GNU scientific library ONCE
# Input: real-valued argument (x)
# Output: Real value
exp_int_once.gsl <- function(x) {
  ei.command <- paste(exp_int_function_filename,x)
  ei <- as.numeric(system(ei.command,intern=TRUE))
  return(ei)
}

