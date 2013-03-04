# Calculate the exponential integral function, for use with the continuous
# powerexp distribution
# Code stolen from numerical recipes and omitted from public distribution
# for obvious copyright reasons!

# The procedure is based on that given in Numerical Recipes, sec. 6.3
# This includes the second (n) argument
# We only ever use the n=1 case here (hence the default argument), but may as
# well be completist
# However (BAD!) this omits ALL sanity-checks on the arguments
# In particular x must be STRICTLY positive

# Compute the exponential integral _once_, using a numerical recipe
# Input: Real-valued argument (x), integer-valued order (n)
# Output: Real value
exp_int_once.nr <- function(x, n=1) {
  maximum_iterations <- 100
  EULER <- 0.5772156649
  FPMIN <- 1.0e-30
  EPS <- 1.0e-7
  n <- 1
  n_minus_one <- n-1   # i.e., zero, but let's preserve the general case
  if (x > 1.0) {
    b <- x+n
    c <- 1.0/FPMIN
    d <- 1.0/b
    h <- d
    for (i in 1:maximum_iterations) {
      a <- -i*(n_minus_one+i)
      b <- b+2.0
      d <- 1.0/(a*d+b)
      c <- b+a/c
      del <- c*d
      h <- h*del
      if(abs(del-1.0) < EPS) {
        return(h*exp(-x))
      }
    }
  } else {
    if (n_minus_one != 0) {
      ans <- 1.0/n_minus_one
    } else {
      ans <- -log(x) - EULER
    } 
    fact <- 1.0
    for (i in 1:maximum_iterations) {
      fact <- fact * (-x/i)
      if (i != n_minus_one) {
        del <- -fact/(i-n_minus_one)
      } else {
	psi <- -EULER + sum(1/(1:n_minus_one))
        # Original code follows in ####
        #### psi <- -EULER
        #### for (j in 1:n_minus_one) {
        ####  psi <- psi + 1.0/j
        #### }
        del <- fact*(-log(x) + psi)
      }
      ans <- ans + del
      if (abs(del) < abs(ans)*EPS) {
        return(ans)
      }
    }
  }
  return(NA) # i.e., something screwed up
}

