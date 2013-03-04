#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Calculate the normalizing factor for the discrete power law with exponential
   cut-off by direct summation */
/* Pr(X=x) \propto x^-a exp(-l*x) */
/* The constant of proportionality is thus the sum of x^-a exp(-l*x) from
   xmin to infinity */
/* This program approximates it very simply by summing from xmin to
   something very large */
/* The analogous procedure for the pure power law is not very efficient (cf.
   zeta_func.c, and the code it calls from the Gnu Scientific Library), but my
   hope is that the exponential factor will make the sum converge more rapidly
   and so fancier math will not be necessary */
/* Only intended to be used with R; not the most elegant integration but it
   may do for now */

/* Three arguments, the scaling exponent, the exponential rate, and the
   lower limit of summation */

#define NUM_TERMS 1000000

int x, xmin; /* Looping index, lower limit of summation */
double a, l; /* Scaling exponent, exponential decay rate */
double norm; /*value of normalizing factor*/
char *program_name; /* name program is invoked under, for errors */

main(int argc, char* argv[]) {
  void usage(void);	/* Warn users about proper usage */

  program_name = argv[0];
  if (argc != 4) {
    usage();
  }
  a = atof(&argv[1][0]);
  if (a <= -1.0) {
    usage();
  }
  l = atof(&argv[2][0]);
  if (l <= 0.0) {
    usage();
  }
  xmin = atoi(&argv[3][0]);
  if (xmin < 1) {
    usage();
  }
  norm = 0.0;
  for(x=xmin;x<xmin+NUM_TERMS;x++) {
    norm += (pow(x,-a) * exp(-x*l));
  }
  printf("%.18e\n",norm);
  return(0);
}

void usage(void) {
  (void) fprintf(stderr, "Usage is %s [floating-point exponent > -1] [floating-point decay rate > 0] [integer lower value >0]\n", program_name);
  exit(8);
}
